# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import pandas as pd
import numpy as np
import scanpy as sc
import os
import subprocess
import shutil
import scipy
import sklearn
import meld
import random
import string
import graphtools as gt


def run_cell_harmony(
    data,
    metadata,
    grouping="sample_labels",
    ctrl_label="ctrl",
    expt_label="expt",
    cell_harmony_path="~/software/cellHarmony-Align/src/cellHarmony_align.py",
    tmp_dir="/tmp/CellHarmony",
    clean_tmp=False,
):
    """
    Runs CellHarmony on a dataset and returns the results file. Returns
    CellHarmony clusters.
    """
    if grouping not in metadata.columns:
        raise ValueError("Could not find {} in metadata".format(grouping))

    if not isinstance(data, pd.DataFrame) or not isinstance(metadata, pd.DataFrame):
        raise ValueError(
            "Both data and metadata must be Pandas DataFrames. Got {} and {}".format(
                type(data), type(metadata)
            )
        )

    if not data.shape[0] == metadata.shape[0]:
        raise ValueError("Both data and metadata must have same axis 0 shape.")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if "~" in cell_harmony_path:
        cell_harmony_path = os.path.expanduser(cell_harmony_path)

    # Create a unique base_dir
    found_free = False
    str_types = string.ascii_uppercase + string.digits
    while found_free is False:
        dir = "".join(random.choice(str_types) for _ in range(16))

        base_dir = os.path.join(tmp_dir, "run_{}".format(dir))
        if os.path.exists(base_dir) is False:
            found_free = True
            os.makedirs(base_dir)

    # Write files for CellHarmony
    data_dirs = []
    for sample in [ctrl_label, expt_label]:
        data_sub = data.loc[metadata[grouping] == sample]
        curr_cells = data_sub.index
        curr_genes = data_sub.columns
        data_sub = scipy.sparse.csc_matrix(data_sub)

        # Directory for current sample
        curr_data_dir = os.path.join(base_dir, sample)
        data_dirs.append(curr_data_dir)
        if not os.path.exists(curr_data_dir):
            os.makedirs(curr_data_dir)

        # Write data to mtx
        scipy.io.mmwrite(os.path.join(curr_data_dir, "matrix.mtx"), data_sub.T)

        # Write genes to file
        genes = pd.DataFrame(curr_genes)
        genes["type"] = "Gene Expression"
        genes.to_csv(
            os.path.join(curr_data_dir, "genes.tsv"),
            sep="\t",
            header=False,
            index=False,
        )

        # Write cells to file
        cells = pd.DataFrame(curr_cells)
        cells.to_csv(
            os.path.join(curr_data_dir, "barcodes.tsv"),
            sep="\t",
            header=False,
            index=False,
        )

    output_file_query = os.path.join(base_dir, "results.txt")
    output_file_ref = os.path.join(base_dir, "results-refererence.txt")

    cmd = [
        "python2",
        cell_harmony_path,
        "-l",
        "-1",
        data_dirs[0],
        data_dirs[1],
        output_file_query,
    ]

    # Run CellHarmony
    subprocess.run(cmd, check=True)

    # Load the results
    results_query = pd.read_csv(output_file_query, delimiter="\t")
    results_ref = pd.read_csv(output_file_ref, delimiter="\t")

    results = pd.concat(
        [results_query, results_ref], axis=0, sort=True, ignore_index=True
    )

    cell_harmony_clusters = np.concatenate(
        [
            results[["Query Barcode", "Ref Partition"]],
            results[["Ref Barcode", "Ref Partition"]],
        ],
        axis=0,
    )

    cell_harmony_clusters = pd.DataFrame(
        cell_harmony_clusters, columns=["Barcode", "CellHarmony"]
    ).drop_duplicates()

    # This should only be run if everything is completed successfully
    if clean_tmp:
        shutil.rmtree(base_dir)

    return cell_harmony_clusters.set_index("Barcode")["CellHarmony"].astype(int)


def find_n_clusters(
    adata, n_clusters, method="louvain", r_start=0.01, r_stop=10, tol=0.001
):
    """
    Helper function to run louvain and leiden clustering through scanpy and
    return a desired number of clusters. Requires scanpy to be installed.
    """

    if method == "louvain":
        cluster_func = sc.tl.louvain
    elif method == "leiden":
        cluster_func = sc.tl.leiden
    else:
        raise ValueError("No such clustering method: {}".format(method))

    if r_stop - r_start < tol:
        cluster_func(adata, resolution=r_start)
        return adata.obs[method].astype(int)

    # Check start
    cluster_func(adata, resolution=r_start)
    n_start = len(np.unique(adata.obs[method]))
    if n_start == n_clusters:
        return adata.obs[method].astype(int)
    elif n_start > n_clusters:
        raise ValueError("r_start is too large. Got: {}".format(r_start))

    # Check end
    cluster_func(adata, resolution=r_stop)
    n_end = len(np.unique(adata.obs[method]))
    if n_end == n_clusters:
        return adata.obs[method].astype(int)
    elif n_end < n_clusters:
        raise ValueError("r_stop is too small. Got: {}".format(r_stop))

    # Check mid
    r_mid = r_start + ((r_stop - r_start) / 2)
    cluster_func(adata, resolution=r_mid)
    n_mid = len(np.unique(adata.obs[method]))
    if n_mid == n_clusters:
        return adata.obs[method].astype(int)

    if n_mid < n_clusters:
        return find_n_clusters(
            adata, n_clusters, method=method, r_start=r_mid, r_stop=r_stop
        )
    else:
        return find_n_clusters(
            adata, n_clusters, method=method, r_start=r_start, r_stop=r_mid
        )


def run_geometry_clustering(curr_adata, curr_metadata, graph, n_clusters):
    """Runs a number of geometry-based clustering algorithms"""
    if not isinstance(curr_adata, sc.AnnData):
        curr_adata = sc.AnnData(curr_adata)
    sc.pp.pca(curr_adata)
    sc.pp.neighbors(curr_adata)
    curr_metadata["Leiden"] = find_n_clusters(curr_adata, n_clusters, "leiden")
    curr_metadata["Louvain"] = find_n_clusters(curr_adata, n_clusters, "louvain")
    curr_metadata["Spectral"] = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed"
    ).fit_predict(graph.K)
    curr_metadata["KMeans"] = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(
        curr_adata.X
    )

    return curr_metadata


class Benchmarker(object):
    def __init__(self, seed=None):
        self.seed = seed
        self.data_phate = None
        self.pdf = None
        self.sample_indicators = None
        self.sample_labels = None
        self.graph = None
        self.graph_kNN = None
        self.meld_op = None
        self.EES = None
        self.estimates = {}

    def set_seed(self, seed):
        self.seed = seed

    def fit_graph(self, data, **kwargs):
        self.graph = gt.Graph(
            data, n_pca=100, use_pygsp=True, random_state=self.seed, **kwargs
        )

    def fit_kNN(self, data, **kwargs):
        self.graph_knn = gt.Graph(
            data,
            n_pca=100,
            kernel_symm=None,
            use_pygsp=True,
            random_state=self.seed,
            **kwargs
        )

    def fit_phate(self, data, **kwargs):
        try:
            import phate
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "PHATE must be installed. Install via pip `pip install --user phate`"
            )

        self.set_phate(phate.PHATE(n_components=3, **kwargs).fit_transform(data))
        return self.data_phate

    def set_phate(self, data_phate):
        if not data_phate.shape[1] == 3:
            raise ValueError("data_phate must have 3 dimensions")
        if not np.isclose(data_phate.mean(), 0):
            # data_phate must be mean-centered
            data_phate = scipy.stats.zscore(data_phate, axis=0)
        self.data_phate = data_phate

    def generate_ground_truth_pdf(self, data_phate=None):
        """Takes a set of PHATE coordinates over a set of points and creates an
        underlying ground truth pdf over the points as a convex combination of
        the input phate coords.
        """
        np.random.seed(self.seed)

        if data_phate is not None:
            self.set_phate(data_phate)
        elif self.data_phate is None:
            raise ValueError(
                "data_phate must be set prior to running generate_ground_truth_pdf()."
            )

        # Create an array of values that sums to 1
        data_simplex = np.sort(np.random.uniform(size=(2)))
        data_simplex = np.hstack([0, data_simplex, 1])
        data_simplex = np.diff(data_simplex)
        np.random.shuffle(data_simplex)

        # Weight each PHATE component by the simplex weights
        sort_axis = np.sum(self.data_phate * data_simplex, axis=1)

        # Pass the weighted components through a logit
        self.pdf = scipy.special.expit(sort_axis)
        return self.pdf

    def generate_sample_labels(self):
        np.random.seed(self.seed)

        # Create sample_labels
        self.sample_indicators = np.random.binomial(1, self.pdf)
        self.sample_labels = np.array(
            ["ctrl" if ind == 0 else "expt" for ind in self.sample_indicators]
        )

    def calculate_EES(self, data=None, **kwargs):
        np.random.seed(self.seed)
        if not self.graph:
            try:
                self.fit_graph(data)
            except NameError:
                raise NameError("Must pass `data` unless graph has already been fit")

        self.meld_op = meld.MELD(**kwargs, verbose=False).fit(self.graph)
        self.EES = self.meld_op.transform(self.sample_labels)
        self.EES = self.EES["expt"].values  # Only keep the expt condition
        self.estimates["EES"] = self.EES
        return self.EES

    def calculate_kNN_average(self, data=None):
        np.random.seed(self.seed)
        if not self.graph_knn:
            try:
                self.fit_kNN(data)
            except NameError:
                raise NameError("Must pass `data` unless graph_nn has already been fit")
        self.EES_knn = (
            np.multiply(
                self.graph_knn.K.toarray(),
                np.tile(self.sample_indicators, self.graph_knn.N).reshape(
                    self.graph_knn.N, -1
                ),
            ).sum(axis=1)
            / self.graph_knn.knn
        )
        self.estimates["kNN"] = self.EES_knn
        return self.EES_knn

    def calculate_graph_average(self, data=None):
        np.random.seed(self.seed)
        if not self.graph:
            try:
                self.fit_graph(data)
            except NameError:
                raise NameError("Must pass `data` unless graph has already been fit")

        self.EES_avg = (
            np.multiply(
                self.graph.K.toarray(),
                np.tile(self.sample_indicators, self.graph.N).reshape(self.graph.N, -1),
            ).sum(axis=1)
            / self.graph.knn
        )
        self.estimates["Graph"] = self.EES_avg
        return self.EES_avg

    def calculate_cluster_average(self, clusters):
        np.random.seed(self.seed)
        cluster_means = pd.Series(self.sample_indicators).groupby(clusters).mean()
        self.Cluster_avg = pd.Series(clusters).apply(lambda x: cluster_means[x])
        self.estimates["Cluster"] = self.Cluster_avg
        return self.Cluster_avg

    def calculate_mse(self, estimate):
        return sklearn.metrics.mean_squared_error(self.pdf, estimate)

    def calculate_r(self, estimate):
        return scipy.stats.pearsonr(self.pdf, estimate)[0]
