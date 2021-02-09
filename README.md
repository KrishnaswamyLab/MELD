# MELD
### Quantifying the effect of experimental perturbations at single-cell resolution


[![Latest PyPi version](https://img.shields.io/pypi/v/MELD.svg)](https://pypi.org/project/MELD/)
![GitHub Actions](https://github.com/KrishnaswamyLab/MELD/workflows/Unit%20Tests/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/MELD/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/MELD?branch=master)
[![Read the Docs](https://img.shields.io/readthedocs/meld-docs.svg)](https://meld-docs.readthedocs.io/)
[![Article](https://zenodo.org/badge/DOI/10.1038/s41587-020-00803-5.svg)](https://doi.org/10.1038/s41587-020-00803-5)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/MELD.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/MELD/)

### Tutorials
For a quick-start tutorial of MELD in Google CoLab, check out this notebook from our [Machine Learning Workshop](https://krishnaswamylab.org/workshop):
* [**MELD Quick Start - Zebrafish data**](https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/exercises/DifferentialAbundance/Answers_Wagner2018_Chordin_Cas9_Mutagenesis.ipynb)

If you're looking for an in-depth tutorial of MELD and VFC, start here:
* [**Guided tutorial in Python - Zebrafish data**](https://nbviewer.jupyter.org/github/KrishnaswamyLab/MELD/blob/master/notebooks/Wagner2018_Chordin_Cas9_Mutagenesis.ipynb).

If you'd like to see how to use MELD without VFC, start here:
* [**Tutorial using MELD without VFC - T cell data**](https://nbviewer.jupyter.org/github/KrishnaswamyLab/MELD/blob/master/notebooks/MELD_thresholding.Tcell.ipynb).

### Introduction

MELD is a Python package for quantifying the effects of experimental perturbations. For an in depth explanation of the algorithm, please read the associated article:

[**Quantifying the effect of experimental perturbations at single-cell resolution**. Daniel B Burkhardt\*, Jay S Stanley\*, Alexander Tong, Ana Luisa Perdigoto, Scott A Gigante, Kevan C Herold, Guy Wolf, Antonio J Giraldez, David van Dijk, Smita Krishnaswamy. Nature Biotechnology. 2021.](https://www.nature.com/articles/s41587-020-00803-5)

The goal of MELD is to identify populations of cells that are most affected by an experimental perturbation. Rather than clustering the data first and calculating differential abundance of samples within clusters, MELD provides a density estimate for each scRNA-seq sample for every cell in each dataset. Comparing the ratio between the density of each sample provides a quantitative estimate the effect of a perturbation at the single-cell level. We can then identify the cells most or least affected by the perturbation.

You can also watch a seminar explaining MELD given by [@dburkhardt](https://github.com/dburkhardt): [![Video](https://img.shields.io/static/v1?label=Zoom&message=Watch%20recording&color=blue&logo=airplay%20video)](https://yale.zoom.us/rec/play/GevmaqSn9xM-j3k3gp1xWnVlKIJGtpUsrv9JZQb5SaqcPfT_pYxwExsXs_jIvIbQsId0eUHw9HkxnvWG.T6ETwk9f0it9LA78?continueMode=true)

### Installation


```
pip install meld
```

### Requirements

MELD requires Python >= 3.6. All other requirements are installed automatically by ``pip``.

### Usage example

```
   import numpy as np
   import meld

   # Create toy data
   n_samples = 500
   n_dimensions = 100
   data = np.random.normal(size=(n_samples, n_dimensions))
   sample_labels = np.random.choice(['treatment', 'control'], size=n_samples)

   # Estimate density of each sample over the graph
   sample_densities = meld.MELD().fit_transform(data, sample_labels)

   # Normalize densities to calculate sample likelihoods
   sample_likelihoods = meld.utils.normalize_densities(sample_densities)
```
