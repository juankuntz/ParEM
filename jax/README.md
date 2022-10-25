# <p align="center">Scalable particle-based alternatives to EM in JAX<br><br>Under review</p>

<div align="center">
  <a href="https://juankuntz.github.io/" target="_blank">Juan&nbsp;Kuntz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://jenninglim.github.io/" target="_blank">Jen Ning&nbsp;Lim</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/" target="_blank">Adam M.&nbsp;Johansen</a> &emsp; </b> 

</div>

## Description

This repository contains Jupyter notebooks illustrating the application of the 
algorithms in [Kuntz et al. (2022)](https://juankuntz.github.io/publication/parem/)
for the Toy Hierarchical Model (Example 1, and Figure 2),
Logistic Regression (Appendix F.4), and Bayesian Neural Network (Section 4.1).
These examples are written in [JAX](https://github.com/google/jax). 
 

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/toy_hierarchical_model.ipynb)  | Toy Hierarchical Model |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_logistic_regression.ipynb) | Bayesian Logistic Regression |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/jax/bayesian_neural_network.ipynb) | Bayesian Neural Network |

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

To setup a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with these packages, clone the repository and use the `environment.yml` file included in it:

```
git clone https://github.com/juankuntz/ParEM.git
conda env create -f ./ParEM/jax/environment.yml
conda activate ParEM
```

Then run the desired notebook:

```
jupyter-notebook ./ParEM/jax/name_of_notebook.ipynb
```

