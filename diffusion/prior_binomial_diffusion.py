import enum
import math

import numpy as np
import torch as th
from torch.distributions.binomial import Binomial

from model.basic_module import mean_flat
from loss.losses import binomial_kl, binomial_log_likelihood, focal_loss

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2),
        )
    elif schedule_name == "alpha_bar_linear":
        beta = []
        for i in range(num_diffusion_timesteps):
            t = i + 1
            beta.append(1/(num_diffusion_timesteps - t + 1))
        return np.array(beta)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_Y = enum.auto()  # the model predicts y_{t-1}
    START_Y = enum.auto()  # the model predicts y_0
    EPSILON = enum.auto()  # the model predicts epsilon


class LossType(enum.Enum):
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    BCE = enum.auto()  # use raw BCE loss
    MIX = enum.auto()  # combine BCE loss and kl loss

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class PriorBinomialDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at 1 and going to T.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    
    def q_mean(self, y_start, p, t):
        """
        Get the distribution q(y_t | y_start, f(x)).

        :param y_start: the [N x C x ...] tensor of noiseless inputs.
        :param p: the output of a segmentor (prior)
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: Binomial distribution parameters, of x_start's shape.
        """
        mean = _extract_into_tensor(self.alphas_cumprod, t, y_start.shape) * y_start + (1 - _extract_into_tensor(self.alphas_cumprod, t, y_start.shape)) * p
        
        return mean

    def q_sample(self, y_start, p, t):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(y_t | y_start, f(x)).

        :param y_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of y_start.
        """

        mean = self.q_mean(y_start, p, t)
        return Binomial(1, mean).sample()
    
    def q_posterior_mean(self, y_start, y_t, p, t):
        """
        Get the distribution q(y_{t-1} | y_t, y_start, f(x))
        """
        assert y_start.shape == y_t.shape

        theta_1 = (_extract_into_tensor(self.alphas, t, y_start.shape) * (1-y_t) + (1 - _extract_into_tensor(self.alphas, t, y_start.shape)) * (th.abs(1-y_t-p))) * (_extract_into_tensor(self.alphas_cumprod, t-1, y_start.shape) * (1-y_start) + (1 - _extract_into_tensor(self.alphas_cumprod, t-1, y_start.shape)) * (1-p))
        theta_2 = (_extract_into_tensor(self.alphas, t, y_start.shape) * y_t + (1 - _extract_into_tensor(self.alphas, t, y_start.shape)) * (th.abs(1-y_t-p))) * (_extract_into_tensor(self.alphas_cumprod, t-1, y_start.shape) * y_start + (1 - _extract_into_tensor(self.alphas_cumprod, t-1, y_start.shape)) * p)

        posterior_mean = theta_2 / (theta_1 + theta_2)

        return posterior_mean
    
    def p_mean(
        self, model, y, t, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(y_{t-1} | y_t, f(x)), as well as a prediction of
        the initial y, i.e., y_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param y: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param denoised_fn: if not None, a function which applies to the
            y_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'pred_ystart': the prediction for y_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = y.shape[0]
        assert t.shape == (B,)
        model_output = model(y, self._scale_timesteps(t), **model_kwargs)

        def process_ystart(y):
            if denoised_fn is not None:
                y = denoised_fn(y)
            return y
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_Y:
            pred_ystart = process_ystart(
                self._predict_ystart_from_yprev(y_t=y, t=t, yprev=model_output)
            )
            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_Y, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_Y:
                pred_ystart = process_ystart(model_output)
            else:
                pred_ystart = process_ystart(
                    self._predict_ystart_from_eps(y_t=y, t=t, eps=model_output)
                )
            model_mean = self.q_posterior_mean(
                y_start=pred_ystart, x_t=y, p=model_kwargs["p"], t=t
            )
            model_mean = th.where((t == 0)[:,None, None, None], pred_ystart, model_mean)
        else:
            raise NotImplementedError(self.model_mean_type)
        return {
            "mean": model_mean,
            "pred_ystart": pred_ystart,
        }
    
    def _predict_ystart_from_eps(self, y_t, t, eps):
        assert y_t.shape == eps.shape
        return (
            th.abs(y_t - eps).to(device=t.device).float()
        )
    
    def _predict_ystart_from_yprev(self, y_t, t, yprev):
        raise NotImplementedError
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def p_sample(
        self, model, y, t, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample y_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param y: the current tensor at y_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param denoised_fn: if not None, a function which applies to the
            y_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_ystart': a prediction of y_0.
        """
        out = self.p_mean(
            model,
            y,
            t,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        sample = Binomial(1, th.clip(out["mean"], min=0, max=1)).sample()
        if t[0] != 0:
            return {"sample": sample, "pred_ystart": out["pred_ystart"]}
        else:
            return {"sample": out["mean"], "pred_ystart": out["pred_ystart"]}
    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param denoised_fn: if not None, a function which applies to the
            y_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = Binomial(1, model_kwargs["p"]).sample().to(device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        y,
        t,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Sample y_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean(
            model,
            y,
            t,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if t[0] != 0:
            alpha_bar_t_1 = _extract_into_tensor(self.alphas_cumprod, t-1, y.shape)
            alpha_bar_t = _extract_into_tensor(self.alphas_cumprod, t, y.shape)
            sigma = (1 - alpha_bar_t_1) / (1 - alpha_bar_t)
            mean = sigma * y + (alpha_bar_t_1 - sigma * alpha_bar_t) * out["pred_ystart"]
            sample = Binomial(1, th.clip(mean, min=0, max=1)).sample()
            return {"sample": sample, "pred_ystart": out["pred_ystart"]}
        else:
            return {"sample": out["mean"], "pred_ystart": out["pred_ystart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = Binomial(1, model_kwargs["p"]).sample().to(device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
    
    def _vb_terms_bpd(
        self, model, y_start, y_t, t, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_ystart': the y_0 predictions.
        """
        true_mean = self.q_posterior_mean(y_start=y_start, y_t=y_t, p=model_kwargs["p"], t=t)
        out = self.p_mean(
            model, y_t, t, model_kwargs=model_kwargs
        )
        kl = binomial_kl(true_mean, out["mean"])

        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -binomial_log_likelihood(y_start, means=out["mean"])
        assert decoder_nll.shape == y_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_ystart": out["pred_ystart"]}

    def training_losses(self, model, y_start, t, model_kwargs=None, lambda_focal=1):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param y_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        y_t = self.q_sample(y_start, model_kwargs["p"], t)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL or self.loss_type == LossType.MIX:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                y_start=y_start,
                y_t=y_t,
                t=t,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
            if self.loss_type == LossType.MIX:
                target = {
                    ModelMeanType.PREVIOUS_Y: self.q_posterior_mean(
                        y_start=y_start, y_t=y_t, p=model_kwargs["p"], t=t
                    ),
                    ModelMeanType.START_Y: y_start,
                    ModelMeanType.EPSILON: self._predict_ystart_from_eps(y_t=y_t, t=t, eps=y_start),
                }[self.model_mean_type]
                model_output = model(y_t, self._scale_timesteps(t), **model_kwargs)
                terms["focal"] = lambda_focal * focal_loss(model_output, target) / np.log(2.0)
                terms["vb"] = terms["loss"]
                terms["loss"] = terms["vb"] + terms["focal"]
        elif self.loss_type == LossType.BCE:
            target = {
                ModelMeanType.PREVIOUS_Y: self.q_posterior_mean(
                    y_start=y_start, y_t=y_t, p=model_kwargs["p"], t=t
                ),
                ModelMeanType.START_Y: y_start,
                ModelMeanType.EPSILON: self._predict_ystart_from_eps(y_t=y_t, t=t, eps=y_start),
            }[self.model_mean_type]
            model_output = model(y_t, self._scale_timesteps(t), **model_kwargs)
            terms["loss"] = mean_flat(-binomial_log_likelihood(target, means=model_output)) / np.log(2.0)
        else:
            raise NotImplementedError(self.loss_type)

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
