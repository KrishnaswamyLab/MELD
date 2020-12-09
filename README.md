# MELD
### Quantifying the effect of experimental perturbations at single cell resolution


[![Latest PyPi version](https://img.shields.io/pypi/v/MELD.svg)](https://pypi.org/project/MELD/)
![GitHub Actions](https://github.com/KrishnaswamyLab/MELD/workflows/Unit%20Tests/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/MELD/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/MELD?branch=master)
[![Read the Docs](https://img.shields.io/readthedocs/meld-docs.svg)](https://meld-docs.readthedocs.io/)
[![bioRxiv Preprint](https://zenodo.org/badge/DOI/10.1101/532846.svg)](https://doi.org/10.1101/532846)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/MELD.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/MELD/)

### Quick Start
* [**Guided tutorial in Python**](https://nbviewer.jupyter.org/github/KrishnaswamyLab/MELD/blob/master/notebooks/Wagner2018_Chordin_Cas9_Mutagenesis.ipynb).

### Introduction

MELD is a Python package for quantifying the effects of experimental perturbations. For an in depth explanation of the algorithm, read our manuscript on BioRxiv.

[**Quantifying the effect of experimental perturbations in single-cell RNA-sequencing data using graph signal processing**. Daniel B Burkhardt\*, Jay S Stanley\*, Ana Luisa Perdigoto, Scott A Gigante, Kevan C Herold, Guy Wolf, Antonio J Giraldez, David van Dijk, Smita Krishnaswamy. BioRxiv. doi:10.1101/532846.](<https://www.biorxiv.org/content/10.1101/532846v2>)

The goal of MELD is to identify populations of cells that are most affected by an experimental perturbation. Rather than clustering the data first and calculating differential abundance of samples within clusters, MELD provides a density estimate for each scRNA-seq sample for every cell in each dataset. Comparing the ratio between the density of each sample provides a quantitative estimate the effect of a perturbation at the single-cell level. We can then identify the cells most or least affected by the perturbation.

### Installation


```
pip install --user meld
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
