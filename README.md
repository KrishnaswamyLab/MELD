## MELD (Manifold Enhancement of Latent Dimensions)


[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/scprep.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/MELD/)

Quantifying the effect of experimental perturbations in scRNA-seq data.

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
