# -----------------------------------------------------------
# This file contains the computation code for FID using `torchmetrics` package.
# See https://torchmetrics.readthedocs.io/en/stable/ for more details.
# -----------------------------------------------------------

import torch

from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(dataset_samples,
                model_samples,
                device='cuda',
                nn_feature=None):
    # Sample images from model and from dataset
    real_images = ((dataset_samples + 1.) / 2 * 255).to(torch.uint8)
    fake_images = ((model_samples + 1.) / 2 * 255).to(torch.uint8)

    if nn_feature is None:
        feature = 2048
        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
            fake_images = fake_images.repeat(1, 3, 1, 1)
    else:
        feature = nn_feature
        real_images = 2 * (real_images.to(torch.float32) / 255. - 0.5)
        fake_images = 2 * (fake_images.to(torch.float32) / 255. - 0.5)

    fid = FrechetInceptionDistance(feature=feature).to(device)
    fid.update(real_images.to(device), real=True)  # Add real images
    fid.update(fake_images.to(device), real=False)  # Add fake images
    # Compute FID
    fid_mean = fid.compute()
    return fid_mean.item()