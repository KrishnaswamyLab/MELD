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
import random, string
import graphtools as gt


def run_cell_harmony(data,
    metadata,
    grouping='RES',
    ctrl_label='ctrl',
    expt_label='expt',
    cell_harmony_path='~/software/cellHarmony-Align/src/cellHarmony_align.py',
    tmp_dir='/tmp/CellHarmony',
    clean_tmp=False):
    '''
    Runs CellHarmony on a dataset and returns the results file. Returns
    CellHarmony clusters.
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
    found_free = False
    str_types = string.ascii_uppercase + string.digits
    while found_free == False:
        dir = ''.join(random.choice(str_types) for _ in range(16))

        base_dir = os.path.join(tmp_dir, 'run_{}'.format(dir))
        if not os.path.exists(base_dir):
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

def run_geometry_clustering(curr_adata, curr_metadata, graph, n_clusters):
    '''Runs a number of geometry-based clustering algorithms'''
    if not isinstance(curr_adata, sc.AnnData):
        curr_adata = sc.AnnData(curr_adata)
    sc.pp.pca(curr_adata)
    sc.pp.neighbors(curr_adata)
    curr_metadata['Leiden'] = find_n_clusters(curr_adata, n_clusters, 'leiden')
    curr_metadata['Louvain'] = find_n_clusters(curr_adata, n_clusters, 'louvain')
    curr_metadata['Spectral'] = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit_predict(graph.K)
    curr_metadata['KMeans'] = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(curr_adata.X)

    return curr_metadata


class Benchmarker(object):
    def __init__(self, seed=None):
        self.seed = seed
        self.data_phate = None
        self.pdf = None
        self.RES_int = None
        self.RES = None
        self.graph = None
        self.graph_kNN = None
        self.meld_op = None
        self.EES = None
        self.estimates = {}
    def set_seed(self, seed):
        self.seed = seed

    def fit_graph(self, data, **kwargs):
        self.graph = gt.Graph(data, n_pca=100, use_pygsp=True, random_state=self.seed,
                              **kwargs)

    def fit_kNN(self, data, **kwargs):
        self.graph_knn = gt.Graph(data, n_pca=100,  kernel_symm=None, use_pygsp=True,
                                random_state=self.seed, **kwargs)

    def generate_ground_truth_pdf(self, data_phate=None):
        '''Takes a set of PHATE coordinates over a set of points and creates an underlying
        ground truth pdf over the points as a convex combination of the input phate coords.
        '''
        np.random.seed(self.seed)

        if data_phate is not None:
            self.data_phate = data_phate

        if not data_phate.shape[1] == 3:
            raise ValueError('data_phate must have 3 dimensions')
        if not np.isclose(data_phate.mean(), 0):
            # data_phate must be mean-centered
            data_phate = scipy.stats.zscore(data_phate, axis=0)

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

    def generate_RES(self):
        np.random.seed(self.seed)

        # Create RES
        self.RES_int = np.random.binomial(1, self.pdf)
        self.RES = np.array(['ctrl' if res == 0 else 'expt' for res in self.RES_int])

    def calculate_EES(self, data=None, **kwargs):
        np.random.seed(self.seed)
        if not self.graph:
            try:
                self.fit_graph(data)
            except NameError:
                raise NameError('Must pass `data` unless graph has already been fit')

        self.meld_op = meld.MELD(**kwargs, verbose=False).fit(self.graph)
        self.EES = self.meld_op.transform(self.RES)
        self.EES = self.EES['expt'].values # Only keep the expt condition
        self.estimates['EES'] = self.EES
        return self.EES

    def calculate_kNN_average(self, data=None):
        np.random.seed(self.seed)
        if not self.graph_knn:
            try:
                self.fit_kNN(data)
            except NameError:
                raise NameError('Must pass `data` unless graph_nn has already been fit')
        self.EES_knn = np.multiply(self.graph_knn.K.toarray(), np.tile(self.RES_int,
                        self.graph_knn.N).reshape(self.graph_knn.N, -1)).sum(axis=1) / self.graph_knn.knn
        self.estimates['kNN'] = self.EES_knn
        return self.EES_knn

    def calculate_graph_average(self, data=None):
        np.random.seed(self.seed)
        if not self.graph:
            try:
                self.fit_graph(data)
            except NameError:
                raise NameError('Must pass `data` unless graph has already been fit')

        self.EES_avg = np.multiply(self.graph.K.toarray(), np.tile(self.RES_int,
                    self.graph.N).reshape(self.graph.N, -1)).sum(axis=1) / self.graph.knn
        self.estimates['Graph'] = self.EES_avg
        return self.EES_avg

    def calculate_cluster_average(self, clusters):
        np.random.seed(self.seed)
        cluster_means = pd.Series(self.RES_int).groupby(clusters).mean()
        self.Cluster_avg = pd.Series(clusters).apply(lambda x: cluster_means[x])
        self.estimates['Cluster'] = self.Cluster_avg
        return self.Cluster_avg

    def calculate_mse(self, estimate):
        return sklearn.metrics.mean_squared_error(self.pdf, estimate)

    def calculate_r(self, estimate):
        return scipy.stats.pearsonr(self.pdf, estimate)[0]

class DatasetGenerator(object):
    def __init__(self, dataset_type, n_cells=None, seed=None):
        self.dataset_type = dataset_type
        if dataset_type == 'three_branch':
            self.generate_dataset = self.generate_three_branch_tree
            self.generate_ground_truth_pdf = self.generate_pdf_three_branch

        else:
            raise NotImplementedError('Currently only dataset_type in ["three_branch"] is supported.')

        self.n_cells = n_cells
        self.seed = seed


    def set_seed(self, seed):
        self.seed = seed

    def set_n_cells(self, n_cells):
        self.n_cells = n_cells

    def generate_three_branch_tree(self):

        # Splatter parameters
        n_groups = 3
        if self.n_cells is not None:

            cells_per_path = self.n_cells
        else:
            cells_per_path = 1000
        params = {'method':'paths', 'batch_cells':cells_per_path * n_groups, 'n_genes':10000,
                  'path_length':[1000], 'group_prob':np.tile(1/n_groups, n_groups),
                  'path_from':[0],#'path_skew':[0.5,0.5],
                 'de_fac_loc':4, 'dropout_type':'binomial', 'verbose':False, 'dropout_prob':0.5}#, 'bcv_common':0.5}

        # Run Splatter
        self.data, self.metadata = self.process_results(scprep.run.SplatSimulate(**params, seed=self.seed))
        return self.data, self.metadata

    def generate_pdf_three_branch(self):
        np.random.seed(self.seed)

        default_p = 0.5
        self.metadata['p'] = default_p

        for path in [1,2,3]:
            default_p = 0.5
            shoulder_width = 0.333
            enrichement = np.random.choice([0.1, 0.9], 1)[0]

            start = np.random.uniform(0.1, 0.54)
            end =  np.random.uniform(0.55, 0.9)

            width = np.round(end - start, 2)

            # Get the enriched region
            self.metadata = self.create_enriched_region(path, start, end, enrichement,
            shoulder_right=False, shoulder_width=shoulder_width, default_p=default_p)
        self.pdf = self.metadata['p']
        return self.pdf

    def generate_branch_and_cluster():
        #Splatter parameters
        n_groups = 2
        cells_per_path = 4000
        params = {'method':'paths', 'batch_cells':cells_per_path * n_groups,
                  'path_length':[1000,3], 'group_prob':[0.4, 0.6],
                  'path_from':[0,0], 'path_nonlinear_prob':0,
                  'de_fac_loc':[1, 1], 'dropout_type':'binomial', 'verbose':False, 'dropout_prob':0.5}

        # Run Splatter
        data, metadata = process_results(scprep.run.SplatSimulate(**params))

        cut_cells = ((metadata['group'] == 2) & (metadata['step'] > 2)) | (metadata['group'] == 1)

        return data[cut_cells], metadata[cut_cells]


    def generate_single_branch():
        # Splatter parameters
        n_groups = 1
        cells_per_path = 4000
        params = {'method':'paths', 'batch_cells':cells_per_path * n_groups, 'n_genes':10000,
                  'path_length':[1000], 'group_prob':[1], 'path_from':[0],
                  'de_fac_loc':3, 'dropout_type':'binomial', 'verbose':False}

        # Run Splatter
        return process_results(scprep.run.SplatSimulate(**params))

    def generate_four_clusters():
        # Splatter parameters
        n_groups = 7
        cells_per_path = 2000
        params = {'method':'paths', 'batch_cells':cells_per_path * n_groups,
                  'path_length':[1000], 'group_prob':np.tile(1/n_groups, n_groups),
                  'path_from':[0,1,1,2,0,0,6],
                  'dropout_type':'binomial', 'verbose':False, 'dropout_prob':0.5}

        # Run Splatter
        data, metadata = process_results(scprep.run.SplatSimulate(**params))

        keep_groups = [3,4,5,7]
        keep_mask = np.isin(metadata['group'], keep_groups)

        data = data.loc[keep_mask]
        metadata = metadata.loc[keep_mask]

        return data, metadata

    def get_shoulder(self, latent_dimension, start, end):
        '''Creates a sine shoulder from `start` height to
        `end` height evaluated over the values in the latent dimension'''
        # Rescale latent dimension to a 0:π interval
        latent_dimension = latent_dimension - np.min(latent_dimension)
        latent_dimension = latent_dimension / np.max(latent_dimension)
        latent_dimension = latent_dimension * np.pi

        sine = np.cos(latent_dimension) * np.sign(start - end)

        sine = sine - np.min(sine)
        sine = sine / np.max(sine)
        sine = sine * np.abs(start-end)
        sine = sine + np.min([start, end])

        return sine

    def get_probability_delta(self, cells_in_window, p, shoulder_left=True, shoulder_right=True, shoulder_width=0.1, default_p=0.5):
        prob_interval = cells_in_window['p'].copy()
        step_interval = cells_in_window['step']

        # This is in terms of steps
        transition_width = (np.max(step_interval) - np.min(step_interval)) * shoulder_width
        try:
            transition_width = int(np.round(transition_width))
        except ValueError:
            raise ValueError(cells_in_window.shape,
                             np.max(step_interval),
                             np.min(step_interval), shoulder_width)

        if shoulder_left:
            # In terms of `step`
            transition_start = np.min(step_interval)
            transition_end = transition_start + transition_width

            # Create a mask over the steps of the path
            shoulder_mask = step_interval <= transition_end

            # Get the shoulder values
            shoulder = self.get_shoulder(step_interval[shoulder_mask], default_p, p)
            prob_interval[shoulder_mask] = shoulder


        if shoulder_right:
            transition_end  = np.max(step_interval)
            transition_start = transition_end - transition_width

            shoulder_mask = step_interval >= transition_start

            shoulder = self.get_shoulder(step_interval[shoulder_mask], p, default_p)
            prob_interval[shoulder_mask] = shoulder
        return prob_interval


    def create_enriched_region(self, path, start, end, p, shoulder_left=True, shoulder_right=True, default_p=0.5, shoulder_width=0.2):
        if start > 1 or end > 1:
            raise ValueError('`start` and `end` must be in range [0,1].')
        metadata = self.metadata.copy()


        curr_cells = self.get_cells_on_path(path)

        step_range = (curr_cells['step'].min(), curr_cells['step'].max())
        n_step = step_range[1] - step_range[0]

        step_start = step_range[0] + (start * n_step)
        step_end = step_range[0] + (end * n_step)

        cells_in_window = self.get_cells_in_window(curr_cells, step_start, step_end)
        if cells_in_window.shape[0] == 0:
            raise ValueError(curr_cells.shape, step_start, step_end, curr_cells['step'].max(), curr_cells['step'].min())
        # Set background p for the window
        cells_in_window['p'] = p

        # Create shoulders
        new_prob = self.get_probability_delta(cells_in_window, p, shoulder_left, shoulder_right,
                                         shoulder_width=shoulder_width, default_p=default_p)
        cells_in_window['p'] = new_prob

        # Assign new group
        curr_cluster = np.max(metadata['true_cluster'])
        cells_in_window['true_cluster'] = curr_cluster + 1

        # Update metadata
        metadata.loc[cells_in_window.index] = cells_in_window
        self.metadata = metadata
        return self.metadata

    def get_cells_in_window(self, metadata, start, end):
        if end < start:
            tmp = start
            start = end
            end = tmp
        return metadata.loc[(start <= metadata['step']) & (metadata['step'] <= end)].copy()

    def get_cells_on_path(self, path):
        return self.metadata.loc[np.isin(self.metadata['group'], path)].copy()

    def process_results(self, results):
        # Assign data to data frame
        data = pd.DataFrame(results['counts'])

        metadata = pd.DataFrame([results['step'], results['group']], index=['step', 'group']).T
        metadata = metadata.astype({'step': int})

        # Sort metadata by step
        metadata = metadata.sort_values(['group', 'step'])

        # Add true_cluster
        metadata['true_cluster'] = 0

        # Sort data
        data = data.loc[metadata.index]

        # Reindex
        new_index = pd.Index(['cell_{}'.format(i) for i in range(metadata.shape[0])])
        data.index = new_index
        metadata.index = new_index

        data_ln = scprep.normalize.library_size_normalize(data)

        return data_ln, metadata

def generate_EES_calc_VFC(seed, data_phate, vfc_op, curr_metadata, n_clusters=3, calc_VFC=True):
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

    np.random.shuffle(data_simplex) # resorts array in place
    sort_axis = np.sum(data_phate * data_simplex, axis=1)
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
    else:
        raise ValueError('n_clusters must be 2 or 3')

    # Create RES
    curr_metadata['RES'] = ['ctrl' if np.random.binomial(1, p) == 0 else 'expt' for p in curr_metadata['p']]

    # Compute EES
    meld_op = meld.MELD().fit(vfc_op.graph)
    EES = meld_op.transform(curr_metadata['RES'])
    curr_metadata['EES'] = EES['expt'].values

    # Calculate VFC
    if calc_VFC:
        vfc_op.transform(meld_op.RES['expt'], meld_op.EES['expt'])
        curr_metadata['VFC'] = vfc_op.predict(n_clusters=n_clusters)

    return curr_metadata


def generate_EES_calc_corr(seed, data_phate, graph, graph_nn, clusters, beta=None, metric='r'):
    '''
    Generates a random experimental signal from a weighted set of PHATE
    coordinates and runs MELD on the RES. Returns a metadata DataFrame.
    '''
    np.random.seed(seed)

    # Set default p
    metadata = pd.DataFrame(generate_ground_truth_pdf(data_phate, seed), columns=['p'])

    # Set cluster asignments
    metadata['clusters'] = clusters

    # Create RES
    metadata['RES_int'] = np.random.binomial(1, metadata['p'])
    metadata['RES'] = ['ctrl' if res == 0 else 'expt' for res in metadata['RES_int']]

    # Compute EES
    meld_op = meld.MELD(beta=beta).fit(graph)
    EES = meld_op.transform(metadata['RES'])
    metadata['EES'] = EES['expt'].values

    # kNN averaging
    metadata['EES_knn'] = np.multiply(graph_nn.K.toarray(), np.tile(metadata['RES_int'], graph_nn.N).reshape(graph_nn.N, -1)).sum(axis=1) / graph_nn.knn

    # Average over MELD graph
    metadata['EES_avg'] = np.multiply(graph.K.toarray(), np.tile(metadata['RES_int'], graph.N).reshape(graph.N, -1)).sum(axis=1) / graph.knn

    # Use clustering
    geno_means = metadata.groupby('clusters')['RES_int'].mean()
    metadata['Cluster_avg'] = metadata['clusters'].apply(lambda x: geno_means[x])

    if metric == 'r':
        metric_meld   =  scipy.stats.pearsonr(metadata['p'], metadata['EES'])[0]
        metric_knn    =  scipy.stats.pearsonr(metadata['p'], metadata['EES_knn'])[0]
        metric_avg    =  scipy.stats.pearsonr(metadata['p'], metadata['EES_avg'])[0]
        metric_clusters =  scipy.stats.pearsonr(metadata['p'], metadata['Cluster_avg'])[0]
    elif metric == 'mse':
        metric_meld   =  sklearn.metrics.mean_squared_error(metadata['p'], metadata['EES'])
        metric_knn    =  sklearn.metrics.mean_squared_error(metadata['p'], metadata['EES_knn'])
        metric_avg    =  sklearn.metrics.mean_squared_error(metadata['p'], metadata['EES_avg'])
        metric_clusters =  sklearn.metrics.mean_squared_error(metadata['p'], metadata['Cluster_avg'])
    else:
        raise ValueError('`metric` must be `r` or `mse`')

    return metric_meld, metric_knn, metric_avg, metric_clusters
