# -----------------------------------------------------------
# This file contains all the utility code for:
# * interacting with the dataset (`get_celeba`, `get_mnist`) modified
#   from https://github.com/enijkamp/short_run_inf.
# * saving checkpoints of the model and algorithm
#   (`save_checkpoint`, `load_checkpoint`).
# * displaying images (`show_images`).
# -----------------------------------------------------------

from typing import Tuple, Optional, Union
import torch
import numpy as np
import PIL
import torchvision.transforms.functional as functional
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os
from torch.utils.data import TensorDataset
from torchvision.utils import make_grid
from torchtyping import TensorType
from pathlib import Path


def show_images(images: TensorType['n_images', 'n_channels', 'width', 'height'],
                show: bool = True,
                path: Optional[Path] = None,
                nrow: int = 1) -> Tuple:
    """Shows and returns figure of image."""
    grid = make_grid(images, nrow=nrow)
    grid = (grid.detach() + 1.) / 2  # Map from [-1, 1] to [0, 1].
    grid = functional.to_pil_image(grid)
    fig = plt.imshow(np.asarray(grid))
    plt.axis('off')
    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()
    return fig, grid


class DatasetWithIndicesAndDetails(TensorDataset):
    """
    Modifies the given TensorDataset class so that (a) it stores image metadata
    and (b) its __getitem__ method returns image, ..., index rather than just
    image, ... (the index is necessary for subsampling in training). The "..."
    is dataset dependent and can be empty (for example, for MNIST it's the
    image labels).
    """

    def __init__(self,
                 *args,  # Tuple of tensors encompassing dataset.
                 n_channels: int = None,  # Number of channels
                 height: int = None,  # Image height
                 width: int = None,  # Image width
                 name: str = ''
                 ):
        super().__init__(*args)
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.name = name

    def __getitem__(self, index):
        return super().__getitem__(index) + (index,)

    def __repr__(self):
        return self.name


def save_checkpoint(algorithm, path: Union[str, Path]) -> None:
    """Saves checkpoint at path."""
    # Check if parent directory exists, if not create it:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(algorithm, "dataset"):
        temp_dataset = algorithm.dataset
        algorithm.dataset = None
    if hasattr(algorithm, "q_batch_index"):
        temp_q_batch_index = algorithm.q_batch_index
        algorithm.q_batch_index = None
    if hasattr(algorithm, "q_batch_index_dl"):
        temp_q_batch_index_dl = algorithm.q_batch_index_dl
        algorithm.q_batch_index_dl = None
    torch.save(algorithm, path)
    if hasattr(algorithm, "dataset"):
        algorithm.dataset = temp_dataset
    if hasattr(algorithm, "q_batch_index"):
        algorithm.q_batch_index = temp_q_batch_index
    if hasattr(algorithm, "q_batch_index_dl"):
        algorithm.q_batch_index_dl = temp_q_batch_index_dl


def load_checkpoint(path: Union[str, Path]):
    """Loads checkpoint from path."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alg = torch.load(path, map_location=torch.device(device))
    return alg


def get_mnist(root_path: Union[str, Path],
              n_images: int,
              width: int = 32,
              height: int = 32,
              train: bool = True):
    """Return the MNIST dataset as `DatasetWithIndicesAndDetails`"""
    dataset = MNIST(root_path, train=train, download=True)

    # Transform dataset into torch tensors with values in [-1, 1] and discard
    # all but the first n_image images.
    transform = transforms.Compose([PIL.Image.fromarray,
                                    transforms.Resize((width, height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5),
                                    ])
    images = torch.stack([transform(dataset.data[i].numpy())
                          for i in range(n_images)])

    tensor_dataset = DatasetWithIndicesAndDetails(images,
                                                  dataset.targets[:n_images],
                                                  n_channels=1,
                                                  width=width,
                                                  height=height,
                                                  name=f'MNIST_{n_images}')
    tensor_dataset.n_classes = 10
    return tensor_dataset


class SingleImagesFolderMTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 cache,
                 transform=None,
                 workers: int = 16,
                 split_size: int = 200,
                 protocol=None,
                 num_images: int = None,
                 name: str = '',
                 train: bool = True):
        self.name = name
        self.n_channels = 3
        if num_images is not None:
            split_size = min(split_size, num_images)
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if transform is not None else lambda \
                    x: x
            self.images = []

            def split_seq(seq, _size):
                newseq = []
                splitsize = 1.0 / _size * len(seq)
                for i in range(_size):
                    newseq.append(seq[int(round(i * splitsize)):int(
                        round((i + 1) * splitsize))])
                return newseq

            def _map(_path_imgs):
                imgs_0 = [self.transform(
                    np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i
                    in _path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.', end='')
                return imgs_1

            path_imgs = os.listdir(root)
            np.random.seed(0)
            n_train = 162770
            n_eval = 202599 - n_train
            if train:
                size = n_train
            else:
                size = n_eval
            idx = np.random.choice(size, size=(num_images,))
            path_imgs.sort()
            offset = 0
            if not train:
                offset = 162771
            if num_images:
                path_imgs = [path_imgs[offset+_id] for _id in idx]

            n_splits = len(path_imgs) // split_size
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(_map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)
            self.height = self.images[0].shape[1]
            self.width = self.images[0].shape[2]

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item]), item

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output

    def __repr__(self):
        return self.name


def get_celeba(data_path: Union[str, Path],
               n_images: int,
               train: bool = True):
    """Return the CelebA dataset as `SingleImagesFolderMTDataset`"""
    dataset = SingleImagesFolderMTDataset(root=data_path,
                                          cache=None,
                                          num_images=n_images,
                                          transform=transforms.Compose([
                                              PIL.Image.fromarray,
                                              transforms.Resize(32),
                                              transforms.CenterCrop(32),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  (0.5, 0.5, 0.5),
                                                  (0.5, 0.5, 0.5)),
                                          ]),
                                          name=f"CelebA_{n_images}",
                                          train=train)
    return dataset
