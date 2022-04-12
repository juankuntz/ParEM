# <p align="center">Scalable particle-based alternatives to EM<br><br> In preparation</p>

<div align="center">
  <a href="https://juankuntz.github.io/" target="_blank">Juan&nbsp;Kuntz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/" target="_blank">Adam M.&nbsp;Johansen</a> &emsp; </b> 

</div>

## Description

This repository contains Jupyter notebooks illustrating the application of the algorithms in [Kuntz & Johansen (2022)](https://juankuntz.github.io/publication/parem/) and reproducing the results in the paper. You can run them either on [Google Colab](https://colab.research.google.com/) or locally on your machine.

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/Toy_Hierarchical_Model.ipynb)  | Toy Hierarchical Model |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/Bayesian_Logistic_Regression.ipynb) | Bayesian Logistic Regression |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juankuntz/ParEM/blob/main/Bayesian_Neural_Network.ipynb) | Bayesian Neural Network |

## Run locally

Running the notebooks locally requires:

- python == 3.9.7
- notebook == 6.4.5
- numpy == 1.20.3
- matplotlib == 3.4.3
- scikit-learn == 0.24.2
- keras == 2.8.0
- jax == 0.2.27
- jaxlib == 0.1.75 

To setup a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with these packages, clone the repository and use the `environment.yml` file included in it:

```
git clone https://github.com/juankuntz/ParEM.git
conda env create -f environment.yml
conda activate ParEM
```

Then run the desired notebook:

```
jupyter-notebook name_of_notebook.ipynb
```

## Citation
If you find the code useful for your research, please consider citing our preprint:

```bib
@article{Kuntz2022,
author = {Kuntz, J. and Johansen, A. M.},
title = {Scalable particle-based alternatives to EM},
journal = {In prepartion},
year  = {2022}
}
```

## License

This work is made available under the MIT License. Please see our main [LICENSE](./LICENSE) file.
