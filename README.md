# <p align="center">Scalable particle-based alternatives to EM<br><br> Preprint</p>

<div align="center">
  <a href="https://juankuntz.github.io/" target="_blank">Juan&nbsp;Kuntz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/" target="_blank">Adam M.&nbsp;Johansen</a> &emsp; </b> 

</div>

## Description

This repository contains Jupyter notebooks illustrating the application of the algorithms in [Kuntz & Johansen (2022)]() and reproducing the results in the paper. You can run them either on [Google Colab](https://colab.research.google.com/) or locally on your machine.

## Run on Colab

The notebooks can be accessed by clicking the links below and logging into a Google Account.

| Link | Example |
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IBW5em23nc-03AYRtsLSJKUrw3zLhyl3?usp=sharing)  | Toy Hierarchical Model |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bb47VSn8u56ONWcixcracwU0Fj-OaD_2?usp=sharing) | Bayesian Logistic Regression |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xcc9iVDS6qo_vNz33aWi8AxPQN9hm7Hf?usp=sharing) | Bayesian Neural Network |

## Run locally

... is built in Python ... using JAX .... Please use the following command to install the requirements:
```shell script
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_releases.html
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

Copyright © 2022, ...

This work is made available under ... Please see our main [LICENSE](./LICENSE) file.
