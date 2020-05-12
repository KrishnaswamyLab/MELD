# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import pandas as pd
import numpy as np
import scanpy as sc
import os
import subprocess
import shutil
import scipy
import sklearn
import scprep
import meld

def run_cell_harmony(
    data,
    metadata,
    grouping='RES',
    ctrl_label='ctrl',
    expt_label='expt',
    cell_harmony_path='~/software/cellHarmony-Align/src/cellHarmony_align.py',
    tmp_dir='/tmp/CellHarmony',
    clean_tmp=False):
    '''
    Runs CellHarmony on a dataset and returns the results file.
    '''
    if not grouping in metadata.columns:
        raise ValueError('Could not find {} in metadata'.format(grouping))

    if not isinstance(data, pd.DataFrame) or not isinstance(metadata, pd.DataFrame):
        raise ValueError('Both data and metadata must be Pandas DataFrames. Got {} and {}'.format(type(data), type(metadata)))

    if not data.shape[0] == metadata.shape[0]:
        raise ValueError('Both data and metadata must have same axis 0 shape.')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if '~' in cell_harmony_path:
        cell_harmony_path = os.path.expanduser(cell_harmony_path)

    # Create a unique base_dir
    max_int = -1
    found_max = False
    while found_max == False:
        max_int += 1
        base_dir = os.path.join(tmp_dir, 'run_{}'.format(max_int))
        if not os.path.exists(base_dir):
            found_max = True
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
        scipy.io.mmwrite(os.path.join(curr_data_dir, 'matrix.mtx'), data_sub.T)

        # Write genes to file
        genes = pd.DataFrame(curr_genes)
        genes['type'] = 'Gene Expression'
        genes.to_csv(os.path.join(curr_data_dir,'genes.tsv'), sep='\t',header=False, index=False)

        # Write cells to file
        cells = pd.DataFrame(curr_cells)
        cells.to_csv(os.path.join(curr_data_dir,'barcodes.tsv'), sep='\t',header=False, index=False)

    output_file_query = os.path.join(base_dir, 'results.txt')
    output_file_ref = os.path.join(base_dir, 'results-refererence.txt')

    cmd = ['python2', cell_harmony_path, '-l', '-1', data_dirs[0], data_dirs[1], output_file_query]

    # Run CellHarmony
    subprocess.run(cmd, check=True)

    # Load the results
    results_query = pd.read_csv(output_file_query, delimiter='\t')
    results_ref = pd.read_csv(output_file_ref, delimiter='\t')

    results = pd.concat([results_query, results_ref], axis=0, sort=True, ignore_index=True)


    cell_harmony_clusters = np.concatenate([results[['Query Barcode', 'Ref Partition']],
               results[['Ref Barcode', 'Ref Partition']]], axis=0)

    cell_harmony_clusters = pd.DataFrame(cell_harmony_clusters,
                                         columns=['Barcode', 'CellHarmony']).drop_duplicates()


    # This should only be run if everything is completed successfully
    if clean_tmp:
        shutil.rmtree(base_dir)

    return cell_harmony_clusters.set_index('Barcode')['CellHarmony'].astype(int)


def find_n_clusters(adata, n_clusters, method='louvain', r_start=0.01, r_stop=10, tol=0.001):
    '''
    Helper function to run louvain and leiden clustering through scanpy and return a desired number
    of clusters. Requires scanpy to be installed.
    '''

    if method == 'louvain':
        cluster_func = sc.tl.louvain
    elif method == 'leiden':
        cluster_func = sc.tl.leiden
    else:
        raise ValueError('No such clustering method: {}'.format(method))

    if r_stop - r_start < tol:
        cluster_func(adata,resolution=r_start)
        return adata.obs[method].astype(int)

    # Check start
    cluster_func(adata,resolution=r_start)
    n_start = len(np.unique(adata.obs[method]))
    if n_start == n_clusters:
        return adata.obs[method].astype(int)
    elif n_start > n_clusters:
        raise ValueError('r_start is too large. Got: {}'.format(r_start))

    # Check end
    cluster_func(adata, resolution=r_stop)
    n_end = len(np.unique(adata.obs[method]))
    if n_end == n_clusters:
        return adata.obs[method].astype(int)
    elif n_end < n_clusters:
        raise ValueError('r_stop is too small. Got: {}'.format(r_stop))

    # Check mid
    r_mid = r_start + ((r_stop - r_start) / 2)
    cluster_func(adata,resolution=r_mid)
    n_mid = len(np.unique(adata.obs[method]))
    if n_mid == n_clusters:
        return adata.obs[method].astype(int)

    if n_mid < n_clusters:
        return find_n_clusters(adata, n_clusters, method=method, r_start=r_mid, r_stop=r_stop)
    else:
        return find_n_clusters(adata, n_clusters, method=method, r_start=r_start, r_stop=r_mid)

def run_geometry_clustering(curr_data, curr_metadata, phate_op, n_clusters):
    '''Runs a number of geometry-based clustering algorithms'''

    curr_adata = sc.AnnData(curr_data)
    sc.pp.pca(curr_adata)
    sc.pp.neighbors(curr_adata)
    curr_metadata['Leiden'] = find_n_clusters(curr_adata, n_clusters, 'leiden')
    curr_metadata['Louvain'] = find_n_clusters(curr_adata, n_clusters, 'louvain')
    curr_metadata['Spectral'] = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit_predict(phate_op.graph.K)
    curr_metadata['KMeans'] = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(curr_data)

    return curr_metadata

def generate_EES_calc_VFC(seed, data_phate, vfc_op, curr_metadata, n_clusters=3):
    '''
    Generates a random experimental signal from a weighted set of PHATE
    coordinates and runs MELD on the RES. Returns a metadata DataFrame.
    '''
    if curr_metadata['phate_clusters'].unique().shape[0] != 60:
        raise ValueError('Expected 60 PHATE clusters, got: {}'.format(
        curr_metadata['phate_clusters'].unique().shape[0]))

    np.random.seed(seed)

    data_simplex = np.sort(np.random.uniform(size=(2)))
    data_simplex = np.hstack([0, data_simplex, 1])
    data_simplex = np.diff(data_simplex)

    ars = [seed]
    mix = data_simplex
    sort_axis = np.sum(data_phate * mix, axis=1)
    curr_metadata['phate_clusters'] = scprep.utils.sort_clusters_by_values(curr_metadata['phate_clusters'],
                                                                  sort_axis)

    # Set default p
    curr_metadata['p'] = 0.5
    curr_metadata['true_clusters'] = 0

    if n_clusters == 2:
        # Assign 1/2 of cells to enriched
        curr_metadata.loc[curr_metadata['phate_clusters'] < 30, 'p'] = 0.9
        curr_metadata.loc[curr_metadata['phate_clusters'] < 30, 'true_clusters'] = 0

        # Assign 1/2 cells to depleted
        curr_metadata.loc[curr_metadata['phate_clusters'] >= 30, 'p'] = 0.1
        curr_metadata.loc[curr_metadata['phate_clusters'] >= 30, 'true_clusters'] = 1
    elif n_clusters == 3:
        # Assign 1/3 of cells to enriched
        curr_metadata.loc[curr_metadata['phate_clusters'] < 20, 'p'] = 0.9
        curr_metadata.loc[curr_metadata['phate_clusters'] < 20, 'true_clusters'] = 1

        # Assign 1/3 cells to depleted
        curr_metadata.loc[curr_metadata['phate_clusters'] > 40, 'p'] = 0.1
        curr_metadata.loc[curr_metadata['phate_clusters'] > 40, 'true_clusters'] = 2

    # Create RES
    curr_metadata['RES'] = ['ctrl' if np.random.binomial(1, p) == 0 else 'expt' for p in curr_metadata['p']]

    # Compute EES
    meld_op = meld.MELD().fit(vfc_op.graph)
    EES = meld_op.transform(curr_metadata['RES'])
    curr_metadata['EES'] = EES['expt'].values

    # Calculate VFC
    vfc_op.transform(meld_op.RES['expt'], meld_op.EES['expt'])
    curr_metadata['VFC'] = vfc_op.predict(n_clusters=n_clusters)

    return curr_metadata
