MELD (Manifold Enhancement of Latent Dimensions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://img.shields.io/pypi/v/MELD.svg
    :target: https://pypi.org/project/MELD/
    :alt: Latest PyPi version
.. image:: https://api.travis-ci.com/KrishnaswamyLab/MELD.svg?branch=master
    :target: https://travis-ci.com/KrishnaswamyLab/MELD
    :alt: Travis CI Build
.. image:: https://coveralls.io/repos/github/KrishnaswamyLab/MELD/badge.svg?branch=master
    :target: https://coveralls.io/github/KrishnaswamyLab/MELD?branch=master
    :alt: Coverage Status
.. image:: https://img.shields.io/readthedocs/meld-docs.svg
    :target: https://meld-docs.readthedocs.io/
    :alt: Read the Docs
.. image:: https://zenodo.org/badge/DOI/10.1101/532846.svg
    :target: https://doi.org/10.1101/532846
    :alt: bioRxiv Preprint
.. image:: https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow
    :target: https://twitter.com/KrishnaswamyLab
    :alt: Twitter
.. image:: https://img.shields.io/github/stars/KrishnaswamyLab/MELD.svg?style=social&label=Stars
    :target: https://github.com/KrishnaswamyLab/MELD/
    :alt: GitHub stars

Quantifying the effect of experimental perturbations in scRNA-seq data.

**Quick Start** - check out our `guided tutorial in python <https://nbviewer.jupyter.org/github/KrishnaswamyLab/MELD/blob/master/notebooks/Wagner2018_Chordin_Cas9_Mutagenesis.ipynb>`_.

For a more in depth explanation of MELD, read our manuscript on BioRxiv:

**Enhancing experimental signals in single-cell RNA-sequencing data using graph signal processing**. Daniel B Burkhardt, Jay S Stanley, Ana Luisa Perdigoto, Scott A Gigante, Kevan C Herold, Guy Wolf, Antonio Giraldez, David van Dijk, Smita Krishnaswamy. `BioRxiv <https://www.biorxiv.org/content/10.1101/532846v1>`__. doi:10.1101/532846.

Installation
============

::

   pip install --user git+git://github.com/KrishnaswamyLab/MELD.git

Requirements
------------

MELD requires Python >= 3.5. All other requirements are installed automatically by ``pip``.

Usage example
=============

::

   import meld
   import graphtools
   G = graphtools.Graph(data, use_pygsp=True)
   meld_score = meld.meld(label, G=G, beta=0.5)
