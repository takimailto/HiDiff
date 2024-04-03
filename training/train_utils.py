import copy
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.optim import AdamW

from . import dist_util
from loggings import logger

from model.fp16_util import zero_grad

from model.basic_module import update_ema
from diffusion.resample import LossAwareSampler, UniformSampler


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        save_path,
        resume_checkpoint,
        prior_model,
        diff_weight,
        schedule_sampler,
        weight_decay,
        max_iterations,
    ):
        self.max_iterations = max_iterations
        self.diff_weight = diff_weight

        self.model = model
        self.prior_model = prior_model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self._load_and_sync_parameters()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.opt_prior = AdamW(self.prior_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        th.autograd.set_detect_anomaly(True)
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())
    
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params
    
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    def run_loop(self):
        data_iter = iter(self.dataloader)
        while (self.step + self.resume_step < self.max_iterations):
            try:
                batch, cond = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch, cond = next(data_iter)
            self.run_step(batch, cond)
            if (self.step + self.resume_step) % self.log_interval == 0:
                logger.dumpkvs()
            if (self.step + self.resume_step) % self.save_interval == 0:
                self.save()
                save_mode_path = os.path.join(self.save_path, f'{(self.step + self.resume_step)}.pth')
                th.save(self.prior_model.state_dict(), save_mode_path)
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            # Save the last checkpoint if it wasn't already saved.
        if (self.step + self.resume_step - 1) % self.save_interval != 0:
            self.save()
            save_mode_path = os.path.join(self.save_path, f'{(self.step + self.resume_step)}.pth')
            th.save(self.prior_model.state_dict(), save_mode_path)
    
    def run_step(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            prior_features, logits, _ = self.prior_model(batch[i : i + self.microbatch].to(dist_util.dev()))
            prior = th.softmax(logits, dim=1)
            detach_micro = {"feature_list": [feature.detach() for feature in prior_features], "p": prior.detach(), "img": batch[i : i + self.microbatch].to(dist_util.dev())}
            micro_cond = cond[i : i + self.microbatch].to(dist_util.dev())

            t, weights = self.schedule_sampler.sample(micro_cond.shape[0], dist_util.dev())
            losses = self.diffusion.training_losses(self.model, micro_cond, t, model_kwargs=detach_micro)
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss.backward()
        self.optimize_normal()

        self.opt_prior.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            prior_features, logits, prior_loss = self.prior_model(batch[i : i + self.microbatch].to(dist_util.dev()))
            prior = th.softmax(logits, dim=1)
            micro_cond = cond[i : i + self.microbatch].to(dist_util.dev())
            micro = {"feature_list": prior_features, "p": prior, "img": batch[i : i + self.microbatch].to(dist_util.dev())}

            t, weights = self.schedule_sampler.sample(micro_cond.shape[0], dist_util.dev())
            losses = self.diffusion.training_losses(self.model, micro_cond, t, model_kwargs=micro)
            loss = (losses["loss"] * weights).mean()
        
            loss_total = prior_loss  + self.diff_weight * loss
            loss_total.backward()
        self.opt_prior.step()

        self.log_step()
        
        lr_ = self.lr * (1.0 - (self.step + self.resume_step ) / self.max_iterations) ** 0.9
        for param_group in self.opt_prior.param_groups:
            param_group['lr'] = lr_
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr_
    
    def optimize_normal(self):
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
