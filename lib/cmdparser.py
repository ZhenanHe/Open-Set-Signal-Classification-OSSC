"""
Command line argument options parser.
Adopted and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import argparse

parser = argparse.ArgumentParser(description='PyTorch Variational Training')

# Dataset and loading
parser.add_argument('--dataset', default='CWRU', help='name of dataset')
parser.add_argument('--known-classes', default='0147', help='name of dataset')
parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--patch-size', default=32, type=int, help='patch size to resize (cwru: 32, nuc: 71/72)')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout probability. '
                                                               'If 0.0 no dropout is applied (default)')
parser.add_argument('-wd', '--weight-decay', default=0.0, type=float, help='Weight decay value (default 0.0)')

# Architecture and weight-init
parser.add_argument('-a', '--architecture', default='WRN', help='model architecture (default: WRN)'
                                                                 'choice of WRN|DCNN|MLP|VGG')

parser.add_argument('--weight-init', default='kaiming-normal',
                    help='weight-initialization scheme (default: kaiming-normal)')
parser.add_argument('--wrn-depth', default=10, type=int,
                    help='amount of layers in the wide residual network (default: 16)')
parser.add_argument('--wrn-widen-factor', default=2, type=int,
                    help='width factor of the wide residual network (default: 8)')
parser.add_argument('--wrn-embedding-size', type=int, default=16,
                    help='number of output channels in the first wrn layer if widen factor is not being'
                         'applied to the first layer (default: 16)')

# Variational parameters
parser.add_argument('--train-var', default=True, type=bool,
                    help='Construct and train variational architecture-KL divergence(default: False)')
parser.add_argument('--var-latent-dim', default=64, type=int, help='Dimensionality of latent space (default: 60)')
parser.add_argument('--var-beta', default=0.5, type=float, help='weight term for KLD loss (default: 0.1)')
parser.add_argument('--var-samples', default=50, type=int,
                    help='number of samples for the expectation in variational training (default: 50)')
parser.add_argument('--model-samples', default=1, type=int,
                    help='Number of stochastic forward inference passes to compute for e.g. MC dropout (default: 1)')

# Training hyper-parameters
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate (default: 1e-3)')
parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization (default 1e-5)')
parser.add_argument('-pf', '--print-freq', default=100, type=int, help='print frequency (default: 100)')

# Resuming training
parser.add_argument('--resume', default='',
                    type=str, help='path to model to load/resume from(default: ''). '
                                   'Also for stand-alone openset outlier evaluation script')

# Open set standalone script
parser.add_argument('--openset-datasets', default='Normal,IR007,IR014,IR021,OR007,OR014,OR021,B007,B014,B021',
                    help='name of openset datasets')

# Open set arguments
parser.add_argument('--distance-function', default='euclidean', help='Openset distance function (default: cosine) '
                                                                     'choice of euclidean|cosine|mix')
parser.add_argument('-tailsize', '--openset-weibull-tailsize', default=0.05, type=float,
                    help='tailsize in percent of data (float in range [0, 1]. Default: 0.05')
