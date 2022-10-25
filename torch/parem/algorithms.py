import torch
from torchtyping import TensorType  # type: ignore
from typing import List
from parem.models import NLVM, NormalVI
import parem.stats as stats
from torch.utils.data import TensorDataset, DataLoader
import parem.utils as utils
import wandb
import torch.distributions as dists


# TODO: Add return type hints throughout project.
# TODO: Add early stopping.

OPTIMIZERS = {'sgd': torch.optim.SGD,
              'adagrad': torch.optim.Adagrad,
              'rmsprop': torch.optim.RMSprop,
              'adam': torch.optim.Adam}


class Algorithm:
    """Prototype class for all algorithms."""

    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 theta_step_size: float = 0.1,
                 q_step_size: float = 1e-3,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu',
                 n_particles: int = 1,
                 ):

        self.train_batch_size = train_batch_size
        self.device = device
        self._model = model
        self.dataset = dataset
        self.q_step_size = q_step_size
        self._theta_optimizer = theta_optimizer
        self.theta_step_size = theta_step_size
        self.n_particles = n_particles
        self._epoch = 0  # Keeps track of the number of training epochs.
        self._losses = []  # List for storing losses incurred during training.

        # Declare theta optimiser
        if type(theta_optimizer) == str:
            self._theta_opt = OPTIMIZERS[theta_optimizer](model.parameters(),
                                                          lr=theta_step_size)
        elif isinstance(theta_optimizer, torch.optim.Optimizer):
            self._theta_opt = theta_optimizer

        # Initialize particles in CPU:
        self._posterior = model.sample_prior(len(dataset), n_particles)
        self._posterior_up_to_date = False  # Flag to update posterior or not.

    def step(self, *args) -> None:
        """Takes a step of the algorithm."""
        raise NotImplementedError()

    def run(self,
            num_epochs: int,
            PATH: str,  # Path to file where checkpoints are to be saved.
            wandb_log=False,
            log_images=True,
            constants_to_be_logged=None,
            compute_stats=False):

        if wandb_log:  # Setup logging.
            wandb.login()
            wandb.init(project=f"PGA_dev_{self.dataset.name}", config=self.__dict__)
            # wandb.watch(self._model, log="all", log_freq=10)

        # Split dataset into batches for training:
        training_batches = torch.utils.data.DataLoader(self.dataset,
                                                       self.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

        # Train:
        for epoch in range(num_epochs):
            avg_loss = 0
            self._model.train()
            for images, *_, idx in training_batches:
                losses = self.step(images.to(device=self.device), idx)
                avg_loss += losses
                print(".", end='')
            avg_loss = avg_loss / len(training_batches)
            self._losses.append(avg_loss)

            # Save checkpoint:
            utils.save_checkpoint(self, PATH)
            if wandb_log:
                from pathlib import Path
                utils.save_checkpoint(self, Path(wandb.run.dir) / f"{epoch}.cpt")
                wandb.save(str((Path(wandb.run.dir) / f"{epoch}.cpt").resolve()))  # wandb.save must take a string.

            # Compute FID and MSE
            self._model.eval()
            stats_dic = {}
            if compute_stats:
                n_samples = 300
                idx = torch.randint(0, len(self.dataset), size=(n_samples,))

                # GMM Sampling
                model_samples = self.synthesize_images(n_samples,
                                                       show=False,
                                                       approx_type='gmm')
                data_samples = torch.stack([self.dataset[id][0] for id in idx], dim=0)
                gmm_fid = stats.compute_fid(data_samples,
                                            model_samples,
                                            n=n_samples,
                                            nn_feature=None)
                gmm_kid = stats.compute_kid(data_samples,
                                            model_samples,
                                            n=n_samples,
                                            nn_feature=None)

                # Standard normal sampling
                model_samples, _ = self._model.sample(n_samples)
                stdg_fid = stats.compute_fid(data_samples,
                                             model_samples,
                                             n=n_samples,
                                             nn_feature=None)
                stdg_kid = stats.compute_kid(data_samples,
                                             model_samples,
                                             n=n_samples,
                                             nn_feature=None)

                mask = torch.ones(32, 32, dtype=torch.bool)

                for i in range(10, 22):
                    for j in range(10, 22):
                        mask[i, j] = False

                idx = torch.randint(0, len(self.dataset), size=(30,))
                img = self.dataset[idx][0]
                reconstructed_image = self.reconstruct(img, mask, show=False)
                mse = ((img - reconstructed_image) ** 2).mean([-1, -2, -3]).mean(0).item()
                stats_dic = {"gmm_fid": gmm_fid,
                             "gmm_kid": gmm_kid,
                             "stdg_fid": stdg_fid,
                             "stdg_kid": stdg_kid,
                             "mse": mse}

            print(f"Epoch {epoch}: Loss {avg_loss:.3f}," + "".join(f" {key} {val:.2f}," for key, val in stats_dic.items()))
            # Log checkpoint:
            if wandb_log:
                if log_images:
                    from wandb import Image
                    std_normal_samples = self.sample_image(10)
                    gmm_samples = self.synthesize_images(10, show=False, approx_type='gmm')
                    wandb.log({"std_normal_samples": Image(std_normal_samples),
                               "gmm_samples": Image(gmm_samples),
                               "loss": avg_loss,
                               ** stats_dic})
                else:
                    wandb.log({"loss": avg_loss, ** stats_dic})
                    
        if wandb_log:
            wandb.finish()

        self.eval()  # Turn on eval mode and switch-off gradients.

    def decode(self, codes: TensorType[..., 'x_dim'], show=False):
        """Returns decoded images stored in codes."""
        self._model.eval()
        decoded_images = self._model(codes.to(self.device)).to(device='cpu')
        if show:
            utils.show_images(decoded_images)
        return decoded_images

    def encode(self,
               images: TensorType[..., 'n_channels', 'width', 'height'],
               mask: TensorType['width', 'height'] = None,
               n_starts: int = 4,  # Number of starts for multi-start optimization.
               patience: int = 50,
               ):
        """
        Returns images encoded. Unless mask is None, it will be applied to
        each image before encoding.
        """
        self.eval()
        if mask is None:
            mask = torch.ones_like(images, dtype=torch.bool)
        else:
            mask = mask.expand(images.shape)

        x = self._model.sample_prior(n_starts, *images.shape[:-3]).requires_grad_(True)
        images = images.expand(n_starts, *images.shape)

        opt = torch.optim.Adam([x], 1.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5)
        mse = torch.nn.MSELoss()
        losses = []
        min_loss = float("Inf")
        for i in range(1000):
            opt.zero_grad()
            loss = mse(images[:, mask], self.decode(x)[:, mask])
            loss.backward()
            opt.step()
            lr_scheduler.step(round(loss.item(), 3))

            # Early stop?
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_ind = i
            if i - min_ind > patience:
                break

        # Find the minimum MSE out of all multistart locations.
        loss = torch.linalg.norm(images[:, mask] - self.decode(x)[:, mask], ord=2, dim=-1)
        ind = loss.argmin()
        return x[ind].clone().detach()

    def sample_image_posterior(self, idx, n: int):  # TODO: add idx type hint
        """Returns n samples from idx's image posterior."""
        raise NotImplementedError()

    def reconstruct(self,
                    images: TensorType['n_images', 'n_channels',
                                       'width', 'height'],
                    mask: TensorType['width', 'height'] = None,
                    show=True,
                    path=None):
        """Returns images reconstructed."""
        self.eval()
        reconstructed_images = self.decode(self.encode(images, mask))
        with torch.no_grad():
            if show:
                for i in range(images.shape[0]):
                    image = images[i]
                    masked_image = image * mask
                    utils.show_images(torch.stack([image, masked_image,
                                                   reconstructed_images[i]]),
                                      path=path / f'{i}.pdf',
                                      nrow=3)
        return reconstructed_images

    def interpolate(self, image1, image2, steps=16, show=True):
        """Returns latent-space interpolations of images 1 and 2."""
        z1 = self.encode(image1)
        z2 = self.encode(image2)
        weights = torch.linspace(0, 1, steps)
        zs = torch.stack([(1 - w)*z1 + w * z2 for w in weights])
        interpolated_images = self.decode(zs)
        if show:
            utils.show_images(interpolated_images)
        return interpolated_images

    def update_posterior(self,):
        """Returns n synthesized images."""
        raise NotImplementedError()

    def loss(self,
             images: TensorType['batch_size', 'n_channels', 'width', 'height'],
             particles: TensorType['batch_size', 'n_particles', 'x_dim']
             ) -> TensorType[()]:
        """
        Returns
        \frac{1}{N|images|}\sum_{n=1}^N\sum_{m in images}
                                            log(p_{\theta_k}(X_k^{n,m}, y^m)).
        """
        log_p = self._model.log_p_v(images, particles)
        return - (1. / images.shape[0]) * log_p.mean()

    def eval(self) -> None:
        """
        Turns on the model's eval mode and disables theta and particle
        gradients.
        """
        self._model.eval()
        self._posterior.requires_grad_(False)
        self._model.requires_grad_(False)

    def sample_image(self, n: int, show=False, path=None):
        """Displays the n images sampled from the joint distribution."""
        samples, _ = self._model.sample(n)
        fig, grid = utils.show_images(samples,
                                      show=show,
                                      path=path,
                                      nrow=int(n ** 0.5))
        return grid

    def synthesize_images(self,
                          n: int = 1,
                          show: bool = True,
                          approx_type: str = 'gaussian',
                          subsample=1000,
                          n_components=150,
                          path=None):
        self.eval()
        self.update_posterior()

        with torch.no_grad():
            if approx_type == 'gaussian':
                mean = torch.mean(self._posterior, [0, 1])
                cov = torch.cov(self._posterior.flatten(0, 1).transpose(0, 1))
                agg_posterior_approx = dists.MultivariateNormal(mean, cov)
                z = agg_posterior_approx.sample(sample_shape=torch.Size([n]))
            elif approx_type == 'gaussian_mixture_labels':
                # Split particles by label:
                particles_by_label = [[] for i in range(10)]
                for i in range(len(self.dataset)):
                    label = int(self.dataset[i][1])
                    particles = self._posterior[i]
                    particles_by_label[label].append(particles)
                particles_by_label = [torch.stack(particles)
                                      for particles in particles_by_label]

                # Build mixture distribution:
                weights, means, covs = [], [], []
                for particles in particles_by_label:
                    weights.append(particles.shape[0])
                    means.append(torch.mean(particles, [0, 1]))
                    covs.append(torch.cov(
                        particles.flatten(0, 1).transpose(0, 1)))
                weights = dists.Categorical(torch.Tensor(weights))
                components = [dists.MultivariateNormal(mean, cov)
                              for mean, cov in zip(means, covs)]

                # Sample distribution:
                z = []
                for i in range(n):
                    label = weights.sample()
                    z.append(components[label].sample())
                z = torch.stack(z)
            elif approx_type == 'gmm':
                from sklearn.mixture import GaussianMixture
                import numpy as np
                best_gmm = None
                lowest_bic = float('Inf')
                if n_components is None:
                    n_components = [10 + 100 * i for i in range(5)]
                else:
                    if type(n_components) == int:
                        n_components = [n_components]
                    elif type(n_components) == list:
                        n_components = n_components
                    else:
                        print("n_components is not int or list of ints")
                        assert 1 == 0
                best_components = 1
                idx = np.random.choice(self._posterior.shape[0], size=subsample)
                x = self._posterior[idx].flatten(0, 1).cpu().numpy()
                for i in n_components:
                    gmm = GaussianMixture(n_components=i).fit(x)
                    bic = gmm.aic(x)
                    if bic < lowest_bic:
                        lowest_bic = bic
                        best_components = i
                        best_gmm = gmm

                weights = dists.Categorical(torch.Tensor(best_gmm.weights_))
                components = [dists.MultivariateNormal(mean, cov)
                              for mean, cov in zip(torch.tensor(best_gmm.means_), torch.tensor(best_gmm.covariances_))]
                z = []
                for i in range(n):
                    label = weights.sample()
                    z.append(components[label].sample())
                z = torch.stack(z).float()

            synthesize_images = self.decode(z)
            if show or (path is not None):
                utils.show_images(synthesize_images, path=path, show=show, nrow=int(n**0.5))
        return synthesize_images


class ParticleBasedAlgorithm(Algorithm):
    """Class from which all particle-based algorithms inherit from."""

    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 theta_step_size: float = 0.1,
                 n_particles: int = 1,
                 particle_step_size: float = 0.1,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        super().__init__(model, dataset, train_batch_size=train_batch_size,
                         theta_step_size=theta_step_size, q_step_size=particle_step_size,
                         theta_optimizer=theta_optimizer, device=device, n_particles=n_particles)

        self.device = device
        self.particle_step_size = particle_step_size

    def sample_image_posterior(self, idx, n: int):  # TODO: add idx type hint
        """Returns first n samples from idx's image posterior."""

        assert n <= self.n_particles, "Number of desired samples cannot be" \
                                      "greater than the number of desired" \
                                      "particles."
        self.eval()
        with torch.no_grad():
            # TODO: Randomize samples?
            image = self.dataset[idx][0].unsqueeze(0)
            posterior_samples = self._model(self._posterior[idx, :n, :]
                                           .to(self.device)
                                           ).detach().to(image.device)
            utils.show_images(torch.concat([image, posterior_samples], dim=0))

    def __repr__(self):
        return "ParticleBasedAlgorithm"


class PGA(ParticleBasedAlgorithm):
    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 lambd: float = 0.1,
                 n_particles: int = 1,
                 particle_step_size: float = 0.1,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        self.lambd = lambd
        theta_step_size = particle_step_size * len(dataset) * lambd
        super().__init__(model, dataset, theta_step_size=theta_step_size,
                         train_batch_size=train_batch_size,
                         n_particles=n_particles,
                         particle_step_size=particle_step_size,
                         theta_optimizer=theta_optimizer,
                         device=device)
        self.name = 'PGA'

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        ## Compute theta gradients ##

        # Turn on theta gradients: 
        self._model.train()
        self._model.requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, self._posterior[idx].to(img_batch.device))

        # Backpropagate theta gradients:
        self._theta_opt.zero_grad()
        loss.backward()

        ## Update particles ##

        # Turn off theta gradients:
        self._model.eval()
        self._model.requires_grad_(False)

        # To avoid exceeding the device's memory, we update the particles
        # using minibatches:
        batches = torch.utils.data.DataLoader(self.dataset, batch_size=750,
                                              pin_memory=True)
        for images, *_, idx in batches:
            # Select particle components to be updated in this iteration:
            sub_particles = (self._posterior[idx].detach().clone()
                             .to(img_batch.device).requires_grad_(True))

            # Compute x gradients:
            log_p_v = self._model.log_p_v(images.to(img_batch.device),
                                         sub_particles).sum()
            x_grad = torch.autograd.grad(log_p_v, sub_particles
                                         )[0].detach().clone()
            del sub_particles

            # Take a gradient step:
            self._posterior[idx] += (self.particle_step_size
                                     * x_grad.to(self._posterior.device))

        # Add noise to all components of all particles:
        self._posterior += ((2 * self.particle_step_size) ** 0.5
                            * torch.randn_like(self._posterior))

        # Update theta:
        self._theta_opt.step()
        self._posterior_up_to_date = False

        # Return value of loss function:
        return loss.item()

    def update_posterior(self,):
        pass


class PGA_smallbatch(ParticleBasedAlgorithm):
    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 lambd: float = 0.1,
                 n_particles: int = 1,
                 particle_step_size: float = 0.1,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        self.lambd = lambd
        theta_step_size = particle_step_size * len(dataset) * lambd
        self.name = "PGA_smallbatch"
        super().__init__(model, dataset, theta_step_size=theta_step_size,
                         train_batch_size=train_batch_size,
                         n_particles=n_particles,
                         particle_step_size=particle_step_size,
                         theta_optimizer=theta_optimizer,
                         device=device)

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        ## Compute theta gradients ##

        # Turn on theta gradients:
        self._model.train()
        self._model.requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, self._posterior[idx].to(img_batch.device))

        # Backpropagate theta gradients:
        self._theta_opt.zero_grad()
        loss.backward()

        ## Update particles ##

        # Turn off theta gradients:
        self._model.eval()
        self._model.requires_grad_(False)

        # To avoid exceeding the device's memory, we update the particles
        # using minibatches:
        batches = torch.utils.data.DataLoader(self.dataset, batch_size=750,
                                              pin_memory=True)

        # Select particle components to be updated in this iteration:
        sub_particles = (self._posterior[idx].detach().clone()
                         .to(img_batch.device).requires_grad_(True))

        # Compute x gradients:
        log_p_v = self._model.log_p_v(img_batch,
                                     sub_particles).sum()
        x_grad = torch.autograd.grad(log_p_v, sub_particles
                                     )[0].detach().clone()
        del sub_particles

        # Take a gradient step:
        self._posterior[idx] += (self.particle_step_size
                                 * x_grad.to(self._posterior.device))

        # Add noise to all components of all particles:
        self._posterior[idx] += ((2 * self.particle_step_size) ** 0.5
                                * torch.randn_like(self._posterior[idx]))

        # Update theta:
        self._theta_opt.step()
        self._posterior_up_to_date = False

        # Return value of loss function:
        return loss.item()

    def update_posterior(self,):
        # To avoid exceeding the device's memory, we update the particles
        # using minibatches:
        pass


class PGA_smallbatchother(ParticleBasedAlgorithm):
    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 lambd: float = 0.1,
                 n_particles: int = 1,
                 particle_step_size: float = 0.1,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        self.lambd = lambd
        theta_step_size = particle_step_size * len(dataset) * lambd
        self.name = "PGA_smallbatchother"
        self.q_batch_index_dl = DataLoader(TensorDataset(torch.tensor(list(range(len(dataset) * n_particles)))),
                                           shuffle=True,
                                           batch_size=train_batch_size * n_particles)
        self.q_batch_index = iter(self.q_batch_index_dl)
        super().__init__(model, dataset, theta_step_size=theta_step_size,
                         train_batch_size=train_batch_size,
                         n_particles=n_particles,
                         particle_step_size=particle_step_size,
                         theta_optimizer=theta_optimizer,
                         device=device)

    def q_reset_iterator(self):
        self.q_batch_index = iter(self.q_batch_index_dl)

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        ## Compute theta gradients ##

        # Turn on theta gradients:
        self._model.train()
        self._model.requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, self._posterior[idx].to(img_batch.device))

        # Backpropagate theta gradients:
        self._theta_opt.zero_grad()
        loss.backward()

        ## Update particles ##

        # Turn off theta gradients:
        self._model.eval()
        self._model.requires_grad_(False)

        # Select particle components to be updated in this iteration:
        idx = next(self.q_batch_index, None)
        if idx is None:
            self.q_reset_iterator()
            idx = next(self.q_batch_index, None)[0]
        else:
            idx = idx[0]

        image_idx, particle_idx = torch.div(idx, self.n_particles, rounding_mode='floor'), idx % self.n_particles  # Map index to image and particle.
        sub_imgs = torch.stack([self.dataset[id][0] for id in image_idx], dim=0).to(img_batch.device)
        sub_particles = (self._posterior[image_idx, particle_idx].detach().clone()
                         .to(img_batch.device).requires_grad_(True))

        # Compute x gradients:
        log_p_v = self._model.log_p(sub_imgs,
                                    sub_particles)
        x_grad = torch.autograd.grad(log_p_v, sub_particles
                                     )[0].detach().clone()

        # Take a gradient step:
        self._posterior[image_idx, particle_idx] += (self.particle_step_size
                                 * x_grad.to(self._posterior.device))

        # Add noise to all components of all particles:
        self._posterior[image_idx, particle_idx] += ((2 * self.particle_step_size) ** 0.5
                                * torch.randn_like(self._posterior[image_idx, particle_idx]))

        # Update theta:
        self._theta_opt.step()
        self._posterior_up_to_date = False

        # Return value of loss function:
        return loss.item()

    def update_posterior(self,):
        # To avoid exceeding the device's memory, we update the particles
        # using minibatches:
        pass


class SubPGA(PGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lags = torch.zeros(self._posterior.shape[0])  # record the number of lags for all particle clouds

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        # Select particle components to be updated in this iteration:
        sub_particles = (self._posterior[idx].detach().clone()
                         .to(img_batch.device).requires_grad_(True))

        # TODO: Use previous model.
        t = self.particle_step_size * self.lags[idx].unsqueeze(-1).unsqueeze(-1)

        # Compute x gradients:
        log_p_v = self._model.log_p_v(img_batch.to(img_batch.device),
                                     sub_particles).sum()

        x_grad = torch.autograd.grad(log_p_v, sub_particles
                                     )[0].detach().clone()
        del sub_particles

        # Take a ULA step:
        self._posterior[idx] += (t.to(self._posterior.device) * x_grad.to(self._posterior.device))

        self._posterior[idx] += (2 * t.to(self._posterior.device)) ** 0.5 * torch.randn_like(self._posterior[idx])

        ## Compute theta gradients ##

        # Turn on theta gradients: 
        self._model.train()
        self._model.requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, self._posterior[idx].to(img_batch.device))

        # Backpropagate theta gradients:
        self._theta_opt.zero_grad()
        loss.backward()

        ## Update particles ##

        # Turn off theta gradients:
        self._model.eval()
        self._model.requires_grad_(False)

        # Select particle components to be updated in this iteration:
        sub_particles = (self._posterior[idx].detach().clone()
                         .to(img_batch.device).requires_grad_(True))

        # Compute x gradients:
        log_p_v = self._model.log_p_v(img_batch.to(img_batch.device),
                                     sub_particles).sum()
        x_grad = torch.autograd.grad(log_p_v, sub_particles
                                     )[0].detach().clone()
        del sub_particles

        # Take a gradient step:
        self._posterior[idx] += (self.particle_step_size
                                 * x_grad.to(self._posterior.device))

        # Add noise to all components of all particles:
        self._posterior[idx] += ((2 * self.particle_step_size) ** 0.5
                                 * torch.randn_like(self._posterior[idx]))
        
        mask = torch.zeros_like(self.lags, dtype=torch.bool)
        mask[idx] = True
        self.lags[~mask] += 1.
        # Update theta:
        self._theta_opt.step()

        # Return value of loss function:
        return loss.item()

    def update_posterior(self,):
        # TODO: to do
        pass


class ShortRun(ParticleBasedAlgorithm):
    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 lambd: float = 0.1,
                 particle_step_size: float = 0.1,
                 n_chain_length: int = 25,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        theta_step_size = particle_step_size * len(dataset) * lambd
        super().__init__(model, dataset, train_batch_size=train_batch_size,
                         theta_step_size=theta_step_size, n_particles=1,
                         particle_step_size=particle_step_size,
                         theta_optimizer=theta_optimizer, device=device)
        self.lambd = lambd
        self.n_chain_length = n_chain_length
        self.particle_step_size = particle_step_size
        self.name = 'ShortRun'

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        ## Run short run ##
        self._model.eval()
        self._model.requires_grad_(False)

        # Initialise particles:
        particles = (torch.randn(img_batch.shape[0], self.n_particles, self._model.x_dim)
                     .to(img_batch.device).requires_grad_(True))

        # Run chain:
        for i in range(self.n_chain_length):
            log_p_v = self._model.log_p_v(img_batch,
                                         particles).sum()
            x_grad = torch.autograd.grad(log_p_v, particles
                                         )[0].detach().clone()

            particles = particles + (self.particle_step_size
                                     * x_grad.to(particles.device))
            particles = particles + ((2 * self.particle_step_size) ** 0.5
                                     * torch.randn_like(particles))
            particles = particles.detach().clone().requires_grad_(True)
        self._posterior[idx] = particles.detach().clone().to(self._posterior.device)

        ## Compute theta gradients ##

        # Turn on theta gradients:
        self._model.train()
        self._model.requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, particles)

        # Backpropagate theta gradients:
        self._theta_opt.zero_grad()
        loss.backward()

        # Update theta:
        self._theta_opt.step()
        self._posterior_up_to_date = False

        # Return value of loss function:
        return loss.item()

    def update_posterior(self,):
        self.eval()

        # Run chain for the whole particle cloud
        if not self._posterior_up_to_date:
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=250)
            for img_particle_batch, *_, particle_idx in dataloader:
                particles = (torch.randn(img_particle_batch.shape[0], self.n_particles, self._model.x_dim)
                             .to(self.device).requires_grad_(True))
                for i in range(self.n_chain_length):
                    log_p_v = self._model.log_p_v(img_particle_batch.to(self.device),
                                                 particles).sum()
                    x_grad = torch.autograd.grad(log_p_v, particles
                                                 )[0].detach().clone()

                    particles = particles + (self.particle_step_size
                                             * x_grad.to(particles.device))
                    particles = particles + ((2 * self.particle_step_size) ** 0.5
                                             * torch.randn_like(particles))
                    particles = particles.detach().clone().requires_grad_(True)
                self._posterior[particle_idx] = particles.detach().clone().to(self._posterior.device)
        self._posterior_up_to_date = True


class AlternatingBackprop(ParticleBasedAlgorithm):
    # A persistent version of short run algorithm.
    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 lambd: float = 0.1,
                 n_chain_length: int = 25,
                 particle_step_size: float = 0.1,
                 theta_optimizer='sgd',  # TODO: add type hint
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        theta_step_size = particle_step_size * len(dataset) * lambd
        super().__init__(model, dataset, theta_step_size=theta_step_size,
                         train_batch_size=train_batch_size,
                         n_particles=1,
                         particle_step_size=particle_step_size,
                         theta_optimizer=theta_optimizer,
                         device=device)
        self.n_chain_length = n_chain_length
        self.name = 'ABP'

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        ## Run short run ##
        self.eval()

        # Run chain for the whole particle cloud
        particles = self._posterior[idx].detach().clone().to(self.device).requires_grad_(True)
        for i in range(self.n_chain_length):
            log_p_v = self._model.log_p_v(img_batch.to(self.device),
                                          particles).sum()
            x_grad = torch.autograd.grad(log_p_v, particles
                                         )[0].detach().clone()

            particles = particles + (self.particle_step_size
                                     * x_grad.to(particles.device))
            particles = particles + ((2 * self.particle_step_size) ** 0.5
                                     * torch.randn_like(particles))
            particles = particles.detach().clone().requires_grad_(True)
        self._posterior[idx] = particles.detach().clone().to(self._posterior.device)
        ## Compute theta gradients ##

        # Turn on theta gradients:
        self._model.train()
        self._model.requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, self._posterior[idx])

        # Backpropagate theta gradients:
        self._theta_opt.zero_grad()
        loss.backward()

        # Update theta:
        self._theta_opt.step()
        self._posterior_up_to_date = False

        # Return value of loss function:
        return loss.item()

    def update_posterior(self):
        pass


class VI(Algorithm):
    def __init__(self,
                 model: NLVM,
                 dataset: utils.TensorDataset,
                 lambd: float = 0.1,
                 n_particles: int = 1,
                 theta_step_size: float = 1e-3,
                 theta_optimizer='sgd',  # TODO: add type hint
                 q_step_size: float = 1e-3,
                 q_optimizer='sgd',
                 train_batch_size: int = 100,
                 device: str = 'cpu'):
        super().__init__(model, dataset, train_batch_size=train_batch_size,
                         theta_step_size=theta_step_size,
                         theta_optimizer=theta_optimizer, device=device, n_particles=n_particles)
        self._encoder = NormalVI(self._model.x_dim, n_in_channel=dataset.n_channels).to(self.device)
        self._encoder_opt = OPTIMIZERS[q_optimizer](self._encoder.parameters(),
                                                   lr=theta_step_size)
        self.name = 'VI'

    def train(self):
        self._model.train()
        self._model.requires_grad_(True)
        self._encoder.train()

    def eval(self):
        self._model.eval()
        self._model.requires_grad_(False)
        self._encoder.eval()

    def step(self,
             img_batch: TensorType['batch_size', 'n_channels',
                                   'width', 'height'],
             idx: TensorType['batch_size']
             ) -> TensorType[()]:
        '''
        Joint gradient updates of the ELBO
        See Eq 7. https://arxiv.org/pdf/1312.6114.pdf
        '''
        self.train()

        # Samples from the posterior
        mu, var = self._encoder(img_batch)
        z = torch.randn(img_batch.shape[0],
                        self.n_particles,
                        self._model.x_dim).to(mu.device) * var.unsqueeze(1) ** 0.5 + mu.unsqueeze(1)

        # Update particle cloud
        # TODO: Should be removed maybe?
        self._posterior[idx] = z.clone().detach().to(self._posterior.device)

        # Compute loss
        log_prob = self._model.log_p_v(img_batch, z).mean()
        kl = 0.5 * (1 + torch.log(var) - mu ** 2 - var).sum()
        # There is an additional multiplicative constant that is the dataset size.
        # This can be removed.
        loss = - (log_prob - kl) * (1. / img_batch.shape[0])

        # Update variational distribution and model.
        self._encoder_opt.zero_grad()
        self._theta_opt.zero_grad()
        loss.backward()
        self._encoder_opt.step()
        self._theta_opt.step()
        self._posterior_up_to_date = False
        return loss.item()

    def sample_image_posterior(self, idx, n: int):  # TODO: add idx type hint
        """Returns first n samples from idx's image posterior."""
        self.eval()
        with torch.no_grad():
            image = self.dataset[idx][0].unsqueeze(0)
            mu, var = self._encoder(image.to(self.device))
            # Sample latent variable
            z = torch.randn(n, self._model.x_dim).to(mu.device) * var ** 0.5 + mu

            posterior_samples = self._model(z.to(self.device)).detach().to(image.device)
            utils.show_images(torch.concat([image, posterior_samples], dim=0))

    def update_posterior(self):
        self.eval()
        # Update particle cloud
        if not self._posterior_up_to_date:
            batches = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=750,
                                                  pin_memory=True)
            for img_batch, *_, idx in batches:
                mu, var = self._encoder(img_batch.to(self.device))
                z = torch.randn(img_batch.shape[0],
                                self.n_particles,
                                self._model.x_dim).to(mu.device) * var.unsqueeze(1) ** 0.5 + mu.unsqueeze(1)
                self._posterior[idx] = z.clone().detach().to(self._posterior.device)
        self._posterior_up_to_date = True
