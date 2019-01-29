import numpy as np

try:
    import pandas as pd
except ImportError:
    pass

try:
    import anndata
except ImportError:
    pass


def is_anndata(data):
    try:
        return isinstance(data, anndata.AnnData)
    except NameError:
        # anndata not installed
        return False


def is_pandas(data, sparse=None):
    try:
        if sparse is None:
            return isinstance(data, pd.DataFrame)
        elif sparse is True:
            return isinstance(data, pd.SparseDataFrame)
        elif sparse is False:
            return isinstance(data, pd.DataFrame) and \
                not isinstance(data, pd.SparseDataFrame)
        else:
            raise ValueError("Expected sparse in [True, False, None]. "
                             "Got {}".format(sparse))
    except NameError:
        # pandas not installed
        return False


def convert_to_same_format(data, target_data, columns=None):
    # create new data object
    if is_pandas(target_data, sparse=True):
        data = pd.SparseDataFrame(data)
        pandas = True
    elif is_pandas(target_data, sparse=False):
        data = pd.DataFrame(data)
        pandas = True
    elif is_anndata(target_data):
        data = anndata.AnnData(data)
        pandas = False
    else:
        # nothing to do
        return data
    # retrieve column names
    target_columns = target_data.columns if pandas else target_data.var
    # subset column names
    try:
        if columns is not None:
            if pandas:
                target_columns = target_columns[columns]
            else:
                target_columns = target_columns.iloc[columns]
    except (KeyError, IndexError, ValueError):
        # keep the original column names
        if pandas:
            target_columns = columns
        else:
            target_columns = pd.DataFrame(index=columns)
    # set column names on new data object
    if pandas:
        data.columns = target_columns
        data.index = target_data.index
    else:
        data.var = target_columns
        data.obs = target_data.obs
    return data


def get_sorting_map(labels, meldscore):
    uniq_clusters = np.unique(labels)
    means = np.array([np.mean(meldscore[labels == cl]) for cl in uniq_clusters])
    new_clust_map = {curr_cl:i for i, curr_cl in enumerate(uniq_clusters[np.argsort(means)])}
    return new_clust_map

def sort_clusters_by_meld_score(clusters, meldscore):
    new_clust_map = get_sorting_map(clusters, meldscore)
    return np.array([new_clust_map[cl] for cl in clusters])
