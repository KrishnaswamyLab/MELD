# MELD (Manifold Enhancement of Latent Dimensions)
### Quantifying the effect of experimental perturbations in scRNA-seq data.


[![Latest PyPi version](https://img.shields.io/pypi/v/MELD.svg)](https://pypi.org/project/MELD/)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/MELD.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/MELD)
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

The goal of MELD is to identify populations of cells that are most affected by an experimental perturbation. Consider a simple two-sample experiment: one experimental sample and one control sample.

The current state of the art is to cluster cells based on global gene expression then 1) calculate the number of cells from each condition in each cluster and 2) do differential expression analysis between conditions within each cluster. The fundamental flaw in this approach is that clusters obtained by global variation are more or less arbitrary partitions of the data. Users can vary parameters to get almost any number of clusters of whatever size they want, so there's no reason to expect that these clusters contain cells that are affected by the perturbation.

MELD solves this problem by introducing the concept of the experimental signal as a *graph signal* on a cell similarity graph. MELD starts using the condition labels that indicate from which sample each cell was measured as the *Raw Experimental Signal* (RES). In signal processing terms, the RES is a noisy indicator of the effect of an experimental perturbation. This is because not only is the biological system and scRNA-seq measurement noisy, but also because an experimental perturbation is not expect to affect all cells in each sample equally. MELD uses methods from Graph Signal Processing to remove this noise to learn an Enhanced Experimental Signal that can be used to 1) identify individual cells that are the most or least enriched across conditions and 2) identify clusters of cells that are both transcriptionally similar are affected by the experimental perturbation to a similar extent.

### Workflow

The basic MELD workflow is:
1. Load, filter, normalize, and transform a counts matrix (or CyTOF data matrix) from each sample
2. Concatenate the matrices from each sample
  * `data, batch_idx = scprep.utils.combine_batches([batch_1, batch_2])`
3. Learn a cell similarity graph `G` using [`graphtools`](https://github.com/KrishnaswamyLab/graphtools)
  * `G = gt.Graph(data)`
4. Visualize data using [`PHATE`](https://github.com/KrishnaswamyLab/PHATE)
5. Create the RES using the batch label for each cell
  * `RES = np.array([-1 if b == '1' else 1 for b in batch_idx])`
6. Filter the RES to recover the EES
  * `EES = MELD().fit_transform(G, RES)`
7. Use the RES and EES to recover Vertex Frequency Clusters
  * `VertexFrequencyCluster(k).fit_predict(G, RES, EES)`

Now you can identify which clusters have the highest or lowest EES values and which genes vary most strongly with the EES.

### Installation


```
pip install --user meld
```

### Requirements

MELD requires Python >= 3.5. All other requirements are installed automatically by ``pip``.

### Usage example

For this usage example to work, you must have already loaded library-size normalized and square root-transformed data into a variable named `data`. You also need to encode the condition of each sample into an `array-like` named `RES` of length `data.shape[0]`. Each entry of the `RES` array should be `1` if the cell is from the experimental condition and `-1` if the cell is from the control condition.

```
   import meld
   import graphtools

   G = graphtools.Graph(data, use_pygsp=True)   
   EES = meld.MELD().fit_transform(G=G, RES=RES)
```
