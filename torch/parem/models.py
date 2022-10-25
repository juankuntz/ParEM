import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


# TODO: maybe there's a better architecture that we should be using?
# TODO: Re-adjust layers to admit different image widths and heights?

class Deterministic(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.gelu):
        super(Deterministic, self).__init__()

        self.activation = activation

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1,
                              padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1,
                               padding=1)

        self.bn = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = out + x  # Skip connection
        return out


class Projection(nn.Module):
    def __init__(self, in_dim, ngf=16, coef=4, activation=F.gelu):
        super(Projection, self).__init__()

        self.activation = activation
        self.ngf = 16
        self.coef = 4

        self.linear = nn.Linear(in_dim, coef * ngf * ngf)
        self.deconv1 = nn.ConvTranspose2d(coef, ngf * coef, kernel_size=5,
                                          stride=1, padding=2, bias=False)
        self.linear_bn = nn.BatchNorm1d(coef * ngf * ngf)
        self.deconv1_bn = nn.BatchNorm2d(ngf * coef)

    def forward(self, x):
        out = self.linear(x)
        out = self.linear_bn(out)
        out = self.activation(out)
        out = out.view(out.size(0), self.coef, self.ngf, self.ngf).contiguous()
        out = self.deconv1(out)
        out = self.deconv1_bn(out)
        out = self.activation(out)
        return out


class Output(nn.Module):
    def __init__(self, x_in, nc):
        super(Output, self).__init__()
        self.output_layer = nn.ConvTranspose2d(x_in, nc, kernel_size=4,
                                               stride=2, padding=1)

    def forward(self, x):
        out = self.output_layer(x)
        out = torch.tanh(out)
        return out


def AnyBatchShape(f):
    def wrapper(self, x):
        not_batch_shape = x.shape[-1]
        batch_shape = x.shape[:-1]
        # Flatten
        x = x.view(-1, not_batch_shape)
        out = f(self, x)
        return out.view(*batch_shape, *out.shape[-3:])
    return wrapper


class NLVM(nn.Module):
    def __init__(self, x_dim=1, nc=3, ngf=16, coef=4, sigma2=1.):
        super(NLVM, self).__init__()
        self.sigma2 = sigma2
        self.x_dim = x_dim
        self.ngf = ngf
        self.nc = nc

        self.projection_layer = Projection(x_dim, ngf=ngf, coef=coef)
        self.deterministic_layer_1 = Deterministic(ngf * coef, ngf * coef)
        self.deterministic_layer_2 = Deterministic(ngf * coef, ngf * coef)
        self.output_layer = Output(ngf * coef, nc)

    @AnyBatchShape
    def forward(self, x):
        out = self.projection_layer(x)
        out = self.deterministic_layer_1(out)
        out = self.deterministic_layer_2(out)
        out = self.output_layer(out)
        return out

    def sample_prior(self, *shape):
        """Draw samples from the (individual-image) prior."""
        device = list(self.parameters())[0].device
        return torch.randn(*shape, self.x_dim, device=device)

    def sample(self, *shape):
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
    def __init__(self, x_dim, n_in_channel=1, n_out_channel=16, n_hidden=512):
        """
        Variational Normal Family with Mean and Variance parameterized. q_\theta(x|y) where
        * y_dim is the dimension of y.
        * x_dim is the dimension of x.
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

    def forward(self, y: TensorType['n_batch', 'n_channels', 'width', 'width'],):
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
