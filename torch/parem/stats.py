import torch
import torch.nn as nn

from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance


def train_nn_feature(dataset,
                     n_hidden: int = 512,
                     n_features: int = 64,
                     n_epoch: int = 500,
                     train: bool = True,
                     nn_type: str = 'conv'):
    # Define Net with feature and classifier components.
    # If we want to use an MLP.
    if nn_type == 'mlp':
        feat = nn.Sequential(nn.Flatten(),
                             nn.Linear(dataset.n_channels * 32 * 32, n_hidden),
                             nn.ReLU(),
                             nn.Linear(n_hidden, n_hidden),
                             nn.ReLU(),
                             nn.Linear(n_hidden, n_features),
                             nn.ReLU())
    elif nn_type == 'conv':
        n_out_channel = 10
        feat = nn.Sequential(nn.Conv2d(dataset.n_channels,
                                       n_out_channel,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2),
                             nn.Conv2d(n_out_channel,
                                       n_out_channel,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1),
                             nn.Flatten(),
                             nn.Linear(32 * 32 * n_out_channel, n_features),
                             nn.ReLU())
    assert(hasattr(dataset, "n_classes"))
    classifier = nn.Linear(n_features, dataset.n_classes)
    net = nn.Sequential(feat, classifier)

    if not train:
        return feat

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    dataloader = torch.utils.data.DataLoader(dataset, 32)

    criterion = nn.CrossEntropyLoss()
    last_improvement = 0
    min_val = torch.inf
    for epoch in range(n_epoch):
        avg_loss = 0.
        for batch, labels, _ in dataloader:
            opt.zero_grad()
            pred = net(batch)
            loss = criterion(pred, labels)
            loss.backward()
            opt.step()
            avg_loss += loss.item()
        avg_loss /= len(dataloader)

        if min_val > avg_loss:
            min_val = avg_loss
            last_improvement = epoch

        if epoch - last_improvement > 10:
            break

    net.eval()
    acc = 0.
    for batch, labels, _ in dataloader:
        pred = torch.softmax(net(batch), dim=-1).argmax(-1)
        batch_acc = (labels == pred).sum() / pred.shape[0]
        acc += batch_acc / len(dataloader)
    print(f"Trained feature NN with accuracy {acc}.")
    return feat


def compute_fid(dataset_samples,
                model_samples,
                device='cuda',
                n=100,
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
    fid_mean = fid.compute()
    return fid_mean.item()


def compute_kid(dataset_samples,
                model_samples,
                device='cuda',
                n=100,
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

    kid = KernelInceptionDistance(feature=feature, subset_size=n).to(device)
    kid.update(real_images.to(device), real=True)  # Add real images
    kid.update(fake_images.to(device), real=False)  # Add fake images
    kid_mean, kid_var = kid.compute()
    return kid_mean.item()