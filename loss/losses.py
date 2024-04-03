import torch as th
import torch
from torch.nn import functional as F

def binomial_kl(mean1, mean2):
    """
    Compute the KL divergence between two Bernoulli.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, mean2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"
    mean1mean2 = th.clamp(mean1/(mean2 + 1e-7), min=1e-7)
    mean1mean2_r = th.clamp((1 - mean1) / (1 - mean2 + 1e-7), min=1e-7)
    return mean1 * th.log(mean1mean2) + (1 - mean1) * th.log(mean1mean2_r)


def binomial_log_likelihood(x, *, means):
    """
    Compute the log-likelihood of a Binomial distribution.

    :param x: the binary mask.
    :param means: the Binomial mean Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    means = th.clamp(means, min=1e-7, max=1-1e-7)
    assert x.shape == means.shape
    log_probs = x * th.log(means) + (1 - x) * (th.log(1 - means))
    assert log_probs.shape == x.shape
    return log_probs

def focal_loss(inputs, targets, gamma=2):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**gamma * BCE_loss
    return F_loss.mean(dim=[1,2,3])