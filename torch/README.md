# <p align="center">Generator network (Section 3.3)<br>

## Description

This repository contains Jupyter notebooks illustrating the application of the 
algorithms in [Kuntz et al. (2023)](https://juankuntz.github.io/publication/parem/)
to the generator network example (Section 3.3). There are two notebooks: one where we train the network on the MNIST dataset and another where we train it on the CelebA dataset. In either case, we use [PyTorch](https://github.com/pytorch/pytorch) and the notebook can be run on Colab or locally on your machine. 

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/MNIST.ipynb)  | Generator network (MNIST) |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/CelebA.ipynb) | Generator network (CelebA) |

## Run locally

The required packages are listed in `requirements.txt`.

To setup a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with these packages, clone the repository and use the `environment.yml` file included here:

```
git clone https://github.com/juankuntz/ParEM.git
conda env create -f ./ParEM/torch/environment.yml
conda activate ParEM_torch
```

Then run the desired notebook:

```
jupyter-notebook ./ParEM/torch/notebooks/name_of_notebook.ipynb
```
