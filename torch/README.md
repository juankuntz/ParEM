# <p align="center">Scalable particle-based alternatives to EM in PyTorch<br><br>Under review</p>

<div align="center">
  <a href="https://juankuntz.github.io/" target="_blank">Juan&nbsp;Kuntz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://jenninglim.github.io/" target="_blank">Jen Ning&nbsp;Lim</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/" target="_blank">Adam M.&nbsp;Johansen</a> &emsp; </b> 

</div>

## Description

This repository contains Jupyter notebooks illustrating the application of the 
algorithms in [Kuntz et al. (2022)](https://juankuntz.github.io/publication/parem/)
for the generator model (Section 4.2).
These examples are written in [PyTorch](https://github.com/pytorch/pytorch). 
We recommend running these examples on Google Colab. 

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/MNIST.ipynb)  | MNIST |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/CelebA.ipynb) | CelebA |

## Run locally

The required packages (and its versions) can be seen in `requirements.txt`.

To setup a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with these packages, clone the repository and use the `environment.yml` file included in it:

```
git clone https://github.com/juankuntz/ParEM.git
conda env create -f ./ParEM/torch/environment.yml
conda activate ParEM_torch
```

Then run the desired notebook:

```
jupyter-notebook ./ParEM/torch/notebooks/name_of_notebook.ipynb
```

