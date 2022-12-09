# <p align="center">Scalable particle-based alternatives to EM<br><br>Under review</p>

<div align="center">
  <a href="https://juankuntz.github.io/" target="_blank">Juan&nbsp;Kuntz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://jenninglim.github.io/" target="_blank">Jen Ning&nbsp;Lim</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/" target="_blank">Adam M.&nbsp;Johansen</a> &emsp; </b> 

</div>

## Description

This repository contains code illustrating the application of the
algorithms in [Kuntz et al. (2022)](https://juankuntz.github.io/publication/parem/)
and reproducing the results in the paper. For the toy hierarchical model (Section 2), Bayesian logistic regression (Section 3.1), and  Bayesian neural network (Section 3.2) examples, we use [JAX](https://github.com/google/jax) and the source code is in the `jax` folder. For the generator network example (Section 3.3), we use [PyTorch](https://pytorch.org/) and the source code is in the `torch` folder.
In either case, the code can be run on [Google Colab](https://colab.research.google.com/) by clicking on the links below, or locally on your machine (see the README.md files in the respective folder for instructions how to do so).

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/toy_hierarchical_model.ipynb)  | Toy hierarchical model |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_logistic_regression.ipynb) | Bayesian logistic regression |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_neural_network.ipynb) | Bayesian neural network |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/MNIST.ipynb) | Generator network (MNIST) |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/CelebA.ipynb) | Generator network (CelebA) |

## Citation
If you find the code useful for your research, please consider citing our preprint:

```bib
@article{Kuntz2022,
author = {Kuntz, J. and Lim, J. N. and Johansen, A. M.},
title = {Scalable particle-based alternatives to EM},
journal = {arXiv preprint arXiv:2204.12965},
year  = {2022}
}
```

## License

This work is made available under the MIT License. Please see our main [LICENSE](./LICENSE) file.
