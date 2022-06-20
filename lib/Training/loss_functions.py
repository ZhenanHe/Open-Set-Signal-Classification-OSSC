import torch
import torch.nn as nn


def var_loss_function(output_samples, target, mu, std, device):
    """
    Computes the loss function consisting of a KL term between approximate posterior and prior and the loss for the
    generative classifier. The number of variational samples is one per default, as specified in the command line parser
    and typically is how VAE models and also our unified model is trained.
    We have added the option to flexibly work with an arbitrary amount of samples.
    """
    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    cl_losses = torch.zeros(output_samples.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples.size(0)):
        cl_losses[i] = class_loss(output_samples[i], target) / torch.numel(target)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)
    # kld = -0.5 * torch.sum(1 + std - mu.pow(2) - std.exp()) / torch.numel(mu)

    return cl, kld
