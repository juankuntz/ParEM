# <p align="center">Scalable particle-based alternatives to EM<br><br>Under review</p>

<div align="center">
  <a href="https://juankuntz.github.io/" target="_blank">Juan&nbsp;Kuntz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://jenninglim.github.io/" target="_blank">Jen Ning&nbsp;Lim</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/" target="_blank">Adam M.&nbsp;Johansen</a> &emsp; </b> 

</div>

## Description

This repository contains the source code illustrating the application of the
algorithms in [Kuntz et al. (2022)](https://juankuntz.github.io/publication/parem/)
and reproducing the results in the paper. For the Toy Hierarchical Model (Example 1, and Figure 2),
Logistic Regression (Appendix F.4), and Bayesian Neural Network (Section 4.1), the code is contained in the
`jax` folder in [JAX](https://github.com/google/jax). As for the generator 
model example (Section 4.2), the code can be found in the `torch` folder.
You can run them either on [Google Colab](https://colab.research.google.com/) or
locally on your machine.

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/toy_hierarchical_model.ipynb)  | Toy Hierarchical Model |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_logistic_regression.ipynb) | Bayesian Logistic Regression |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_neural_network.ipynb) | Bayesian Neural Network |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/mnist.ipynb) | Generator Network: MNIST |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/torch/notebooks/celeba.ipynb) | Generator Network: CelebA |

## Citation
If you find the code useful for your research, please consider citing our preprint:

```bib
@article{Kuntz2022,
author = {Kuntz, J. and Johansen, A. M.},
title = {Scalable particle-based alternatives to EM},
journal = {arXiv preprint arXiv:2204.12965},
year  = {2022}
}
```

## License

This work is made available under the MIT License. Please see our main [LICENSE](./LICENSE) file.
