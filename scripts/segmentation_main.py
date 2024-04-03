import sys
sys.path.append("..")
import argparse

import torch as th

from loggings import logger
from training import dist_util

from datasets.synapseloader import Synapsedataset
from datasets.bratsloader import BRATSDataset
from datasets.cvcloader import CVCDataset
from datasets.kvasirloader import KvasirDataset
from datasets.driveloader import DriveDataset
from datasets.chaseloader import CHASEDataset

from diffusion.resample import create_named_schedule_sampler

from external.merit.networks import MERIT_Cascaded

from script_util import (
    create_model_and_diffusion,
    add_dict_to_argparser,
)
from training.train_utils import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating data loader...")
    if args.data_name == "Synapse":
        ds = Synapsedataset(args.base_dir, args.list_dir)
        args.img_channels = 1
        args.K = 9
    elif args.data_name == 'BRATS':
        ds = BRATSDataset(args.list_dir, test_flag=False)
        args.img_channels = 4
        args.K = 4
    elif args.data_name == "Kvasir":
        ds = KvasirDataset(args.img_path, args.depth_path)
        args.img_channels=3
        args.K=1
    elif args.data_name == "CVC":
        ds = CVCDataset(args.img_path, args.depth_path)
        args.img_channels=3
        args.K=1
    elif args.data_name == "Drive":
        ds = DriveDataset(args.img_path, args.depth_path)
        args.img_channels=3
        args.K=1
    elif args.data_name == "CHASE":
        ds = CHASEDataset(args.img_path, args.depth_path)
        args.img_channels=3
        args.K=1

    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,)
    
    data = iter(datal)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        img_channels=args.img_channels,
        num_channels=args.num_channels,
        condition_dim_list=args.condition_dim_list,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        timestep_respacing=args.timestep_respacing,
        ltype=args.ltype,
        mean_type=args.mean_type,
        rescale_timesteps=args.rescale_timesteps,
        model_type=args.model_type,
        K=args.K,
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    prior_model = MERIT_Cascaded(n_class=args.K, img_size_s1=(256, 256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    prior_model.load_state_dict(th.load(args.prior_path))
    prior_model.to(dist_util.dev())
    prior_model.train()

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        save_path=args.save_path,
        diff_weight = args.diff_weight,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        prior_model=prior_model,
        max_iterations=args.max_iterations,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name="Synapse",
        base_dir="",
        list_dir="",
        img_path="",
        depth_path="",
        batch_size=32,
        num_channels=128,
        condition_dim_list=[],
        diffusion_steps=10,
        noise_schedule="linear",
        timestep_respacing="",
        ltype="mix",
        mean_type="epsilon",
        rescale_timesteps=False,
        model_type="binary_cross_merit_real",
        schedule_sampler="uniform",
        prior_path="",
        microbatch=4,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=1000,
        resume_checkpoint="",
        save_path="",
        weight_decay=0.0,
        max_iterations=40000,
        gpu_dev="0",
        diff_weight=0.0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
