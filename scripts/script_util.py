import argparse

from diffusion import prior_binomial_diffusion as pbd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.BBDM_binary import UNetModel as UNetModel_binary
from model.BBDM_real import UNetModel  as UNetModel_real



def create_model_and_diffusion(
    img_channels,
    num_channels,
    condition_dim_list,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    ltype,
    mean_type,
    rescale_timesteps,
    model_type="binary_cross_merit",
    K=9,
):
    model = create_model(
        K,
        img_channels,
        condition_dim_list,
        model_type,
        num_channels,
    )

    diffusion = create_priorbinomial_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        ltype=ltype,
        mean_type=mean_type,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    K,
    img_channels,
    condition_dim_list,
    model_type,
    num_channels,
):
    out_channels=K
    
    in_channels = img_channels + out_channels * 2
    
    if model_type == "binary_cross_merit":
        model = UNetModel_binary(
            out_channels=out_channels,
            in_channels=in_channels,
            condition_dim_list=condition_dim_list,
            model_channels=num_channels,
        )
    elif model_type == "binary_cross_merit_real":
        model = UNetModel_real(
            out_channels=out_channels,
            in_channels=in_channels,
            condition_dim_list=condition_dim_list,
            model_channels=num_channels,
        )

    else:
        raise NotImplementedError(f"unknown ModelType: {model_type}")

    return model

def create_priorbinomial_diffusion(
    steps=1000,
    noise_schedule="linear",
    ltype="bce",
    mean_type="ystart",
    rescale_timesteps=False,
    timestep_respacing="",
):
    betas = pbd.get_named_beta_schedule(noise_schedule, steps)
    if ltype == "rescale_kl":
        loss_type = pbd.LossType.RESCALED_KL
    elif ltype == "kl":
        loss_type = pbd.LossType.KL
    elif ltype == "bce":
        loss_type = pbd.LossType.BCE
    elif ltype == "mix":
        loss_type = pbd.LossType.MIX
    else:
        raise NotImplementedError(f"unknown LossType: {ltype}")
    if not timestep_respacing:
        timestep_respacing = [steps]
    if mean_type == "ystart":
        model_mean = pbd.ModelMeanType.START_Y
    elif mean_type == "epsilon":
        model_mean = pbd.ModelMeanType.EPSILON
    elif mean_type == "previous":
        model_mean = pbd.ModelMeanType.PREVIOUS_Y
    else:
        raise NotImplementedError(f"unknown ModelMeanType: {mean_type}")

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
