# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
import graphtools as gt
import meld
import pygsp
import unittest

from scipy import sparse
from utils import make_batches, assert_warns_message, assert_raises_message
from nose.tools import assert_raises

from packaging import version

def test_benchmarker_density():
    np.random.seed(0)
    data = np.random.normal(0, 2, (200, 200))

    benchmarker = meld.Benchmarker(seed=0)

    benchmarker.fit_phate(data, verbose=False)

    benchmarker.generate_ground_truth_pdf()

    benchmarker.generate_sample_labels()
    benchmarker.calculate_MELD_likelihood(data=data) #implicitly tests graph
    sample_likelihoods_mse = benchmarker.calculate_mse(benchmarker.sample_likelihoods)

    np.testing.assert_allclose(4.962076e-05, sample_likelihoods_mse)

    # Test mean-center works
    benchmarker.set_phate(benchmarker.data_phate + 1)

    # Test with data_phate passed
    benchmarker.generate_ground_truth_pdf(
        benchmarker.data_phate
    )

def test_benchmarker_set_params():
    benchmarker = meld.Benchmarker()
    benchmarker.set_seed(0)

    with assert_raises_message(
        ValueError,
        "data_phate must have 3 dimensions"
    ):
        benchmarker.set_phate(np.random.normal(0, 2, (10, 2)))

def test_benchmarker_calc_pdf_before_fit_phate():
    benchmarker = meld.Benchmarker()

    with assert_raises_message(
        ValueError,
        "data_phate must be set prior to running generate_ground_truth_pdf()."
    ):
        benchmarker.generate_ground_truth_pdf()

def test_benchmarker_calc_MELD_without_graph_or_data():
    benchmarker = meld.Benchmarker()
    with assert_raises_message(
        NameError,
        "Must pass `data` unless graph has already been fit"
    ):
        benchmarker.calculate_MELD_likelihood()
