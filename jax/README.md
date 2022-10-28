# <p align="center">Toy hierarchical model (Section 2.1), Bayesian neural network (Section 4.1), and Bayesian logistic regression (Appendix F.4)<br>

## Description

This folder contains Jupyter notebooks illustrating the application of the 
algorithms in [Kuntz et al. (2022)](https://juankuntz.github.io/publication/parem/)
to the toy hierarchical model (Section 2.1), Bayesian neural network (Section 4.1), and Bayesian logistic regression (Appendix F.4) examples.
These examples are written in [JAX](https://github.com/google/jax) and they either be run on Colab or locally on your machine. 
 

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/toy_hierarchical_model.ipynb)  | Toy hierarchical model |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_logistic_regression.ipynb) | Bayesian logistic regression |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_neural_network.ipynb) | Bayesian neural network |

## Run locally

Running the notebooks locally requires:

- jax == 0.2.27
- jaxlib == 0.1.75 
- keras == 2.8.0
- matplotlib == 3.4.3
- notebook == 6.4.5
- numpy == 1.20.3
- python == 3.9.7
- scikit-learn == 0.24.2
- wget == 3.2

To setup a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with these packages, clone the repository and use the `environment.yml` file included here:

```
git clone https://github.com/juankuntz/ParEM.git
conda env create -f ./ParEM/jax/environment.yml
conda activate ParEM_jax
```

Then run the desired notebook:

```
jupyter-notebook ./ParEM/jax/name_of_notebook.ipynb
```

