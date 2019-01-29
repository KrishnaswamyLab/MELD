MELD (Manifold Enhancement of Latent Dimensions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://img.shields.io/pypi/v/MELD.svg
    :target: https://pypi.org/project/MELD/
    :alt: Latest PyPi version
.. image:: https://api.travis-ci.com/KrishnaswamyLab/MELD.svg?branch=master
    :target: https://travis-ci.com/KrishnaswamyLab/MELD
    :alt: Travis CI Build
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

Note, this repository is under active development. Please check back on
Monday Feb 4th 2019 for Version 0.1.

For now, check out our preprint on BioRxiv:

.. raw:: html <a href="https://www.biorxiv.org/content/10.1101/532846v1"><strong>Enhancing experimental signals in single-cell RNA-sequencing data using graph signal processing</strong><br />Daniel B Burkhardt, Jay S Stanley, Ana Luisa Perdigoto, Scott A Gigante, Kevan C Herold, Guy Wolf, Antonio Giraldez, David van Dijk, Smita Krishnaswamy. <em>BioRxiv.</em>doi:10.1101/532846</a>

Installation
============

::

   pip install --user git+git://github.com/KrishnaswamyLab/MELD.git

Requirements
------------

MELD requires Python >= 3.5. All other requirements are installed automatically by ``pip``.

Optional
--------

`pyunlocbox <https://pyunlocbox.readthedocs.io/en/stable/>`__ is used for fast solving via proximal splitting. Install it via ``pip install pyunlocbox``.

Usage example
=============

::

   import meld
   import graphtools
   G = graphtools.Graph(data, use_pygsp=True)
   meld_score = meld.meld(label, gamma=0.5, g=G)
