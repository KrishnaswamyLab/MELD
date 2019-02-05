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

Quantifying the effect of experimental perturbations in scRNA-seq data.

Note, this repository is under active development. Please check back on
Monday Feb 4th 2019 for Version 0.1.

.. toctree::
    :maxdepth: 2

    installation
    reference

Quick Start
===========

You can use `meld` with your single cell data as follows::

   import meld
   import graphtools
   G = graphtools.Graph(data, use_pygsp=True)
   meld_score = meld.meld(label, G=G, beta=0.5)

Help
====

If you have any questions or require assistance using MELD, please contact us at https://krishnaswamylab.org/get-help
