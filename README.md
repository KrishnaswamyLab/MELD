## MELD (Manifold Enhancement of Latent Dimensions)

[![bioRxiv Preprint](https://zenodo.org/badge/DOI/10.1101/532846.svg)](https://www.biorxiv.org/content/10.1101/532846v1)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/scprep.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/MELD/)

Quantifying the effect of experimental perturbations in scRNA-seq data. 

Note, this repository is under active development. Please check back on Monday Feb 4th 2019 for Version 0.1. For now, check out our preprint on BioRxiv: 
  
[**Enhancing experimental signals in single-cell RNA-sequencing data using graph signal processing**  
Daniel B Burkhardt, Jay S Stanley, Ana Luisa Perdigoto, Scott A Gigante, Kevan C Herold, Guy Wolf, Antonio Giraldez, David van Dijk, Smita Krishnaswamy. *BioRxiv.* doi: https://doi.org/10.1101/532846](https://www.biorxiv.org/content/10.1101/532846v1)

## Installation

```
pip install --user git+git://github.com/KrishnaswamyLab/MELD.git#subdirectory=python
```

## Requirements
1. pygsp
	Install via
	`pip install pygsp`
2. graphtools
	https://github.com/KrishnaswamyLab/graphtools
	Used for importing data, building graphs, and getting gradient matrices.


## Optional

1. pyunlocbox
	https://pyunlocbox.readthedocs.io/en/stable/
	Used for fast solving via proximal splitting
	Install via
	`pip install pyunlocbox`


## Usage example

```
import meld
import graphtools
G = graphtools.Graph(data, use_pygsp=True)
meld_score = meld.meld(label, gamma=0.5, g=G)
```
