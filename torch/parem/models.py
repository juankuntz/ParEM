# -----------------------------------------------------------
# This file contains implementation of the model used in Section 4.2, and its layers.
# The `class` for the layers: `Deterministic`, `Projection`, `Output` is
# modified from https://github.com/enijkamp/short_run_inf.
# The `class` for the model: `NLVM` is based upon https://arxiv.org/abs/1912.01909.
# The `class` called `NormalVI` is the variational approximation used as a baseline for Section 4.2.
# See Appendix F.3 https://arxiv.org/pdf/2204.12965.pdf for more details.
# -----------------------------------------------------------

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class Deterministic(nn.Module):
    """
    The Deterministic Layer used in NLVM.
    """
    def __init__(self, in_dim: int, out_dim: int, activation=F.gelu):
        super(Deterministic, self).__init__()

        self.activation = activation

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1,
                              padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1,
                               padding=1)

        self.bn = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x: TensorType[..., 'n_channels', 'in_dim1', 'in_dim2']
                ) -> TensorType[...,  'n_channels', 'out_dim1', 'out_dim2']:
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = out + x  # Skip connection
        return out


class Projection(nn.Module):
    """
    The Projection Layer used in NLVM.
    """
    def __init__(self,
                 in_dim: int,
                 ngf: int = 16,
                 coef: int = 4,
                 activation=F.gelu):
        super(Projection, self).__init__()

        self.activation = activation
        self.ngf = 16
        self.coef = 4

        self.linear = nn.Linear(in_dim, coef * ngf * ngf)
        self.deconv1 = nn.ConvTranspose2d(coef, ngf * coef, kernel_size=5,
                                          stride=1, padding=2, bias=False)
        self.linear_bn = nn.BatchNorm1d(coef * ngf * ngf)
        self.deconv1_bn = nn.BatchNorm2d(ngf * coef)

    def forward(self, x: TensorType[..., 'in_dim']) -> TensorType[..., 'n_channels', 'out_dim1', 'out_dim2']:
        out = self.linear(x)
        out = self.linear_bn(out)
        out = self.activation(out)
        out = out.view(out.size(0), self.coef, self.ngf, self.ngf).contiguous()
        out = self.deconv1(out)
        out = self.deconv1_bn(out)
        out = self.activation(out)
        return out


class Output(nn.Module):
    """
    The Output Layer used in NLVM.
    """
    def __init__(self,
                 x_in: int,
                 nc: int):
        super(Output, self).__init__()
        self.output_layer = nn.ConvTranspose2d(x_in, nc, kernel_size=4,
                                               stride=2, padding=1)

    def forward(self, x: TensorType[..., 'n_channels', 'in_dim1', 'in_dim2']
                ) -> TensorType[...,  'n_channels', 'out_dim1', 'out_dim2']:
        out = self.output_layer(x)
        out = torch.tanh(out)
        return out


def anyBatchShape(f):
    """
    Wrap the function to allow for different ndim for the batch size.
    """
    def wrapper(self, x: TensorType[..., 'in_dim']) -> TensorType[..., 'in_dim']:
        not_batch_shape = x.shape[-1]
        batch_shape = x.shape[:-1]
        # Flatten
        x = x.view(-1, not_batch_shape)
        out = f(self, x)
        return out.view(*batch_shape, *out.shape[-3:])
    return wrapper


class NLVM(nn.Module):
    """
    Implementation of the model. Similar to https://github.com/enijkamp/short_run_inf.
    """
    def __init__(self,
                 x_dim: int = 1,
                 nc: int = 3,
                 ngf: int = 16,
                 coef: int = 4,
                 sigma2: int = 1.):
        super(NLVM, self).__init__()
        self.sigma2 = sigma2
        self.x_dim = x_dim
        self.ngf = ngf
        self.nc = nc

        self.projection_layer = Projection(x_dim, ngf=ngf, coef=coef)
        self.deterministic_layer_1 = Deterministic(ngf * coef, ngf * coef)
        self.deterministic_layer_2 = Deterministic(ngf * coef, ngf * coef)
        self.output_layer = Output(ngf * coef, nc)

    @anyBatchShape
    def forward(self, x: TensorType[..., 'x_dim']) -> TensorType[..., 'n_channels', 'out_dim1', 'out_dim2']:
        out = self.projection_layer(x)
        out = self.deterministic_layer_1(out)
        out = self.deterministic_layer_2(out)
        out = self.output_layer(out)
        return out

    def sample_prior(self, *shape: int):
        """Draw samples from the (individual-image) prior."""
        device = list(self.parameters())[0].device
        return torch.randn(*shape, self.x_dim, device=device)

    def sample(self, *shape: int):
        """Draw samples from the joint distribution."""
        latent = self.sample_prior(*shape)
        obs = self.forward(latent)
        obs = obs  # + self.sigma2 ** 0.5 * torch.randn_like(obs, device=obs.device)
        obs.clip_(-1., 1.)
        return obs, latent

    def log_p(self,
              image: TensorType['n_batch', 'n_channels', 'height', 'width'],
              x: TensorType['n_batch', 'x_dim']) -> TensorType[()]:
        # Log prior
        log_prior = - 0.5 * (x ** 2).sum([])

        # Log likelihood
        x_decoded = self.forward(x)
        log_likelihood = - 0.5 * ((image - x_decoded) ** 2 / self.sigma2).sum()
        return log_prior + log_likelihood

    def log_p_v(self,
                image: TensorType['n_batch', 'n_channels', 'height', 'width'],
                x: TensorType['n_batch', 'n_particles', 'x_dim']
                ) -> TensorType['n_particles']:
        # Log prior
        log_prior = - 0.5 * (x ** 2).sum([0, -1])

        # Log likelihood
        x_decoded = self.forward(x)
        log_likelihood = - 0.5 * ((image.unsqueeze(1) - x_decoded) ** 2
                                  / self.sigma2).sum([0, -3, -2, -1])
        return log_prior + log_likelihood


class NormalVI(nn.Module):
    """
    Implementation of the Normal Variational family.
    """
    def __init__(self,
                 x_dim: int,
                 n_in_channel: int = 1,
                 n_out_channel: int = 16,
                 n_hidden: int = 512):
        """
        :param x_dim: Dimension of the latent variable.
        :param n_in_channel: number of channels of the images.
        :param n_out_channel: number of channel output of the conv layer.
        :param n_hidden: dimension of the hidden (linear) layer.
        """
        super().__init__()
        self.x_dim = x_dim
        self.conv1 = nn.Conv2d(n_in_channel,
                               n_out_channel,
                               kernel_size=3,
                               stride=1,
                               padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_out_channel,
                               n_out_channel * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * n_out_channel * 2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, x_dim * 2)

    def forward(self, y: TensorType['n_batch', 'n_channels', 'width', 'width']
                ) -> Tuple[TensorType['n_batch', 'x_dim'], TensorType['n_batch', 'x_dim']]:
        y = self.conv1(y)
        y = F.relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.pool2(y)
        y = y.flatten(start_dim=1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        mu = y[..., :self.x_dim]
        var = F.softplus(y[..., self.x_dim:])
        return mu, var + 1e-3
