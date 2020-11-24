===========================================================================
meld
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/meld/"><img src="https://img.shields.io/pypi/v/meld.svg" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://travis-ci.com/KrishnaswamyLab/meld"><img src="https://api.travis-ci.com/KrishnaswamyLab/meld.svg?branch=master" alt="Travis CI Build"></a>

.. raw:: html

    <a href="https://coveralls.io/github/KrishnaswamyLab/MELD?branch=master"><img src="https://coveralls.io/repos/github/KrishnaswamyLab/MELD/badge.svg?branch=master" alt="Coverage Status"></img></a>

.. raw:: html

    <a href="https://meld-docs.readthedocs.io/"><img src="https://img.shields.io/readthedocs/meld-docs.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://doi.org/10.1101/532846"><img src="https://zenodo.org/badge/DOI/10.1101/532846.svg" alt="bioRxiv Preprint"></a>

.. raw:: html

    <a href="https://meld-docs.readthedocs.io/"><img src="https://img.shields.io/readthedocs/meld-docs.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><img src="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" alt="Twitter"></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/meld/"><img src="https://img.shields.io/github/stars/KrishnaswamyLab/meld.svg?style=social&label=Stars" alt="GitHub stars"></a>

MELD is a Python package for quantifying the effects of experimental perturbations. For an in depth explanation of the algorithm, read our manuscript on BioRxiv: https://www.biorxiv.org/content/10.1101/532846v2

The goal of MELD is to identify populations of cells that are most affected by an experimental perturbation. Rather than clustering the data first and calculating differential abundance of samples within clusters, MELD provides a density estimate for each scRNA-seq sample for every cell in each dataset. Comparing the ratio between the density of each sample provides a quantitative estimate the effect of a perturbation at the single-cell level. We can then identify the cells most or least affected by the perturbation.

.. toctree::
    :maxdepth: 2

    installation
    reference

Quick Start
===========

You can use `meld` as follows::

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

Help
====

If you have any questions or require assistance using MELD, please contact us at https://krishnaswamylab.org/get-help
