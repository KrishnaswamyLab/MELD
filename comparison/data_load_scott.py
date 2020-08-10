import numpy as np
from scipy.io import loadmat
import pandas as pd
import phate
from sklearn import datasets, cluster
import fcsparser
import h5py
import os
import scprep

from .embed import PCA


def load_wishbone():
    meta, data = fcsparser.parse(
        # "/data/lab/DataSets/Setty_Wishbone/wishbone_thymus_panel1_rep1.fcs")
        os.path.join("/home/scottgigante", "data", "datasets",
                     "Setty_Wishbone", "wishbone_thymus_panel1_rep1.fcs"))
    data = data[np.concatenate([data.columns[:13], data.columns[14:21]])]
    data = np.arcsinh(data / 5)
    return np.array(data), np.array(data["CD3"]).reshape(-1), False, "CD3", "Wishbone", {"k": 60, "t": 12}  # NOQA: E501


def load_dla_tree(n_dim=50, **kwargs):
    data, color = phate.tree.gen_dla(n_dim=n_dim, **kwargs)
    colors, labels = pd.factorize(color)
    return data, colors, True, labels, "DLA Tree", {"t": 150, 'a': None}


def load_swiss_roll(noise=0.05, sigma=0.7):
    X, t = datasets.make_swiss_roll(n_samples=1500, noise=noise)
    X = X + np.random.normal(0, sigma, X.shape)
    return X, t, False, "Time", "Swiss Roll", {'t': 22, 'a': None}


def load_frey():
    comparison_figures_rows, comparison_figures_cols = 28, 20
    ff = loadmat(os.path.join("/home/scottgigante", "data", "datasets",
                              "frey_rawface.mat"),
                 squeeze_me=True,
                 struct_as_record=False)
    ff = ff["ff"].T.reshape(
        (-1, comparison_figures_rows * comparison_figures_cols))
    ff = PCA(ff, n_components=100)
    return ff, np.arange(len(ff)), False, "Time", "Frey Faces", {"a": 20, "k": 5, "t": 500}  # NOQA: E501


def load_teapot():
    x = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "tea.mat"))
    data = x["Input"][0, 0][0].transpose().reshape(-1, 3, 101, 76).transpose(
        0, 3, 2, 1) / 255
    data = data.reshape(len(data), -1)
    data = PCA(data, n_components=100)
    return data, np.arange(len(data)), False, "Time", "Teapot", {'k': 10, 't': 26, 'a': None}  # NOQA: E501


def load_artificial_tree0():
    data = loadmat(os.path.expanduser(
        "~/data/datasets/phate/newTree_noiseless.mat"))
    colors, labels = pd.factorize(data["C"].reshape(-1))
    return data["M"], colors, True, labels, "Artificial Tree sigma=0", {}


def load_artificial_tree3():
    data = loadmat(os.path.expanduser(
        "~/data/datasets/phate/newTree_sig3.mat"))
    colors, labels = pd.factorize(data["C"].reshape(-1))
    return data["M"], colors, True, labels, "Artificial Tree sigma=3", {"a": 5, "k": 17, "t": 800}  # NOQA: E501


def load_artificial_tree7():
    data = loadmat(os.path.expanduser(
        "~/data/datasets/phate/newTree_sig7.mat"))
    colors, labels = pd.factorize(data["C"].reshape(-1))
    return data["M"], colors, True, labels, "Artificial Tree sigma=7", {"a": 10, "k": 18, "t": 250}  # NOQA: E501


def load_artificial_tree11():
    data = loadmat(os.path.expanduser(
        "~/data/datasets/phate/newTree_sig11.mat"))
    colors, labels = pd.factorize(data["C"].reshape(-1))
    return data["M"], colors, True, labels, "Artificial Tree sigma=11", {"a": 10, "k": 18, "t": 250}  # NOQA: E501


def load_artificial_tree20():
    data = loadmat(os.path.expanduser(
        "~/data/datasets/phate/newTree_sig20.mat"))
    colors, labels = pd.factorize(data["C"].reshape(-1))
    return data["M"], colors, True, labels, "Artificial Tree sigma=20", {"a": 10, "k": 20, "t": 20}  # NOQA: E501


def load_half_circles():
    data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "halfCircles.mat"))
    colors, labels = pd.factorize(data["C"].reshape(-1))
    return data["M"], colors, True, labels, "Half Circles", {"a": 10, "k": 7, "t": 150, "alpha_decay": True}  # NOQA: E501


def load_gmm_connected():
    data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "gmmPHATE_connected.mat"))
    # data = loadmat(os.path.join("/home/scottgigante", "data",
    # "datasets","gmmConnected_big.mat"))
    colors, labels = pd.factorize(data["labels"].reshape(-1))
    data = data["data"]
    newdata = data[colors == 0] + \
        np.random.permutation(data[colors == 0]) - \
        np.random.permutation(data[colors == 3]) + \
        6 * np.mean(data[colors == 0], axis=0) - \
        6 * np.mean(data[colors == 1], axis=0)
    data = np.vstack([data, newdata])
    colors = np.concatenate([colors, np.repeat(4, len(newdata))])
    labels = np.concatenate([labels, [5]]).astype(np.int32)
    return data, colors, True, labels, "Connected GMM old", {"a": 10, "k": 4, "t": 10}  # NOQA: E501


def load_gmm():
    data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "gmmPhate.mat"))
    colors, labels = pd.factorize(data["labels"].reshape(-1))
    return data["data"], colors, True, labels, "Connected GMM", {"a": 10, "k": 5, "t": 2}


def load_bonemarrow():
    clusters = pd.read_csv(os.path.join(
        "/home/scottgigante", "data", "datasets", "bmmc", "MAP.csv"), header=None)
    clusters.columns = pd.Index(['wells', 'clusters'])
    bmmsc = pd.read_csv(
        os.path.join("/home/scottgigante", "data", "datasets",
                     "bmmc", "BMMC_myeloid.csv.gz"), index_col=0)
    bmmsc_norm = scprep.normalize.library_size_normalize(bmmsc)
    bmmsc_norm = np.sqrt(bmmsc_norm)
    bmmsc_norm = PCA(bmmsc_norm, n_components=100)
    # using cluster labels from original publication
    C = np.array(clusters['clusters']).reshape(-1)
    colors, labels = pd.factorize(C)
    labels = np.array(['Erythrocyte C1',
                       'Erythrocyte C2',
                       'Erythrocyte C3',
                       'Erythrocyte C4',
                       'Erythrocyte C5',
                       'Erythrocyte C6',
                       'Early Erythrocyte C7',
                       'Megakaryocyte C8',
                       'Early Neutrophil C9',
                       'Early Monocyte C10',
                       'Dendritic Cells C11',
                       'Early Basophil C12',
                       'Basophil C13',
                       'Monocyte C14',
                       'Monocyte C15',
                       'Neutrophil C16',
                       'Neutrophil C17',
                       'Eosinophil C18',
                       'Lymphoid Progenitors (NK) C19'])[labels - 1]
    return bmmsc_norm, colors, True, labels, "Bone Marrow", {"k": 4, "a": 100, "t": 20, "alpha_decay": True}  # NOQA: E501


def load_velten():
    rna_data = pd.read_csv(os.path.join(
        "/home/scottgigante", "data", "datasets",
        "velten", "GSE75478_transcriptomics_raw_filtered_I1.csv"),
        header=0, sep=',', index_col=0).T
    facs_data = pd.read_csv(os.path.join(
        "/home/scottgigante", "data", "datasets",
        "velten", "GSE75478_transcriptomics_facs_indeces_filtered_I1.csv"),
        header=0, sep=',', index_col=0).T
    facs_data = facs_data.iloc[np.where(facs_data["FACS_Lin"] > -1000)[0]]
    cells = list(set(rna_data.index).intersection(facs_data.index))
    rna_data = rna_data.loc[cells]
    # color_col = np.argwhere(rna_data.columns == 'KEL (ENSG00000197993)')[0,
    # 0]
    libnorm_data = np.log(
        0.1 + scprep.normalize.library_size_normalize(rna_data.values))
    clusters = cluster.AgglomerativeClustering(n_clusters=8).fit_predict(
        libnorm_data[:, np.argpartition(-np.std(libnorm_data, axis=0),
                                        1000)[:1000]])
    rna_data = scprep.normalize.library_size_normalize(rna_data.T).T
    rna_data = np.sqrt(rna_data)
    # color = rna_data[:, color_col].reshape(-1)
    rna_data = PCA(rna_data, n_components=100)
    colors, labels = pd.factorize(clusters)
    return rna_data, colors, True, labels, "Velten HSC", {"k": 3, "a": 20, "t": 25}  # NOQA: E501


def load_EB():
    EB_data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "EB", "EBdata.mat"))
    time = EB_data['cells'].reshape(-1)
    data_norm = scprep.normalize.library_size_normalize(
        EB_data['data'].toarray())
    data_norm = scprep.transform.sqrt(data_norm)
    data_norm = PCA(data_norm, n_components=100)
    # time = [1,1,2,2,3,3,4,4,5,5]
    time = np.array([0, 7, 13.5, 20, 27])[time - 1]
    return data_norm, time, False, "Day", "EB", {"k": 12, "a": 10, "t": 24, "alpha_decay": True}  # NOQA: E501


def load_ziesel():
    data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "StenData.mat"))["data"]
    # C = loadmat(
    #     os.path.join("/home/scottgigante", "data", "datasets",
    #                  "StenDataClusters.mat"))["C"].reshape(-1)
    lab = pd.read_csv(os.path.expanduser("~/data/datasets/StenDataLabels.tsv"),
                      index_col=0,
                      header=None, sep="\t", low_memory=False).T
    lab = lab.set_index("cell_id")
    C = lab['level1class']
    data = np.sqrt(data)
    data = PCA(data, n_components=100)
    colors, labels = pd.factorize(C)
    return data, colors, True, labels, "Ziesel Neuronal", {"a": 10, "k": 5, "t": 35, "alpha_decay": True}  # NOQA: E501


def load_ipsc():
    data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "ipscData.mat"))
    time = data["data_time"].reshape(-1)
    data = data["data"]
    # time = [1,1,2,2,...,11,11,12,12]
    time = np.array([0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])[time - 1]
    return data, time, False, "Day", "iPSC", {"k": 15, "t": 250, 'a': None}


def load_coil():
    import imageio  # noqa
    N = 20
    M = 72
    imsize = 128 * 128
    X = np.zeros((N * M, imsize))
    counter = 0
    for i in range(N):
        for j in range(M):
            X[counter, :] = imageio.imread(os.path.join(
                "/home/scottgigante", "data", "datasets",
                "COIL20", "img", "obj{}__{}.png".format(
                    i + 1, j))).flatten()
            counter += 1
    labels = ['Duck', 'Blocks1', 'Car1', 'Cat', 'Anacin', 'Car2',
              'Blocks2', 'BabyPowder', 'Tylenol', 'Vaseline',
              'Blocks3', 'Cup', 'Piggybank', 'Ashtray', 'Trashcan',
              'Conditioner', 'Bowl', 'Teacup', 'Car3', 'CreamCheese']
    labels = np.repeat(labels, M)
    # time = np.tile(np.arange(M), N)
    X = PCA(X, n_components=100)
    colors, labels = pd.factorize(labels)
    return X, colors, True, labels, "COIL20", {'k': 3, 'a': 105, 't': 37}  # NOQA: E501


def load_MNIST():
    mnist = datasets.fetch_openml('mnist_784')
    X = PCA(mnist.data, n_components=100)
    colors, labels = pd.factorize(mnist.target.reshape(-1))
    labels = labels.astype(np.int32)
    return X, colors, True, labels, "MNIST", {'k': 5, 'a': None, 't': 105}


def load_retinal_bipolar():
    clusters = pd.read_csv(
        os.path.join("/home/scottgigante", "data", "datasets",
                     "shekhar_retinal_bipolar",
                     "retina_clusters.tsv"), sep="\t", index_col=0)
    cells = pd.read_csv(
        os.path.join("/home/scottgigante", "data",
                     "datasets",
                     "shekhar_retinal_bipolar", "retina_cells.csv"),
        header=None,
        index_col=False).values.reshape(-1)[:-1]
    with h5py.File(os.path.join("/home/scottgigante", "data", "datasets",
                                "shekhar_retinal_bipolar",
                                "retina_data.mat"), 'r') as f:
        data = pd.DataFrame(
            np.array(f['data']).T,
            index=cells)
    merged_data = pd.merge(data, clusters, how='left',
                           left_index=True, right_index=True)
    merged_data = merged_data.loc[~np.isnan(merged_data['CLUSTER'])]
    data = merged_data[merged_data.columns[:-2]]
    data = scprep.normalize.library_size_normalize(data.values)
    data = np.sqrt(data)
    data = PCA(data, n_components=100)
    clusters, labels = pd.factorize(
        merged_data[merged_data.columns[-1]])
    # labels = ['11', '23', '5', '4', '1', '3', '10', '6', '16_1', '2', '13', '14', '7',
    #   '12', '15_1', '9', '18', '17', '8', '15_2', '24', '21', '19', '16_2',
    #   '20', '22', '25', '26']
    cluster_assign = {
        '1': 'Rod BC',
        '2': 'Muller Glia',
        '7': 'BC1A',
        '9': 'BC1B',
        '10': 'BC2',
        '12': 'BC3A',
        '8': 'BC3B',
        '14': 'BC4',
        '3':  'BC5A',
        '13': 'BC5B',
        '6': 'BC5C',
        '11': 'BC5D',
        '5': 'BC6',
        '4': 'BC7',
        '15_1': 'BC8/9_1',
        '15_2': 'BC8/9_2',
        '16_1':  'Amacrine_1',
        '16_2':  'Amacrine_2',
        '20': 'Rod PR',
        '22': 'Cone PR',
    }
    labels = np.array(labels)
    for label, celltype in cluster_assign.items():
        labels = np.where(labels == label, celltype, labels)
    return data, clusters, True, labels, "Retinal Bipolar", {'k': 15, 'a': None, 't': 40}  # NOQA: E501


def load_haber():
    data = loadmat(os.path.join(
        "/home/scottgigante", "data", "datasets", "intenstine_data.mat"))
    clusters = data["meta3"].reshape(-1)
    clusters = np.array([c[0] for c in clusters])
    clusters, labels = pd.factorize(clusters)
    data = data["data"]
    data = PCA(data, 50)
    return data, clusters, True, labels, "Haber Intenstine", {'k': 15, 't': 50, 'a': None}  # NOQA: E501


def splatter(method="paths",
             nBatches=1, batchCells=100,
             nGenes=10000,
             batch_facLoc=0.1, batch_facScale=0.1,
             mean_rate=0.3, mean_shape=0.6,
             lib_loc=11, lib_scale=0.2, lib_norm=False,
             out_prob=0.05,
             out_facLoc=4, out_facScale=0.5,
             de_prob=0.1, de_downProb=0.1,
             de_facLoc=0.1, de_facScale=0.4,
             bcv_common=0.1, bcv_df=60,
             dropout_type='none', dropout_prob=0.5,
             dropout_mid=0, dropout_shape=-1,
             group_prob=1,
             path_from=0, path_length=100, path_skew=0.5,
             path_nonlinearProb=0.1, path_sigmaFac=0.8,
             seed=None):
    if seed is None:
        seed = np.random.randint(2**16 - 1)
    if dropout_type == 'binomial':
        dropout_type = "none"
    else:
        dropout_prob = None
    np.random.seed(seed)
    splat = scprep.run.RFunction(
        setup="""
            library(splatter)
            library(scater)
            library(magrittr)
        """,
        args="""
            method='paths',
            nBatches=1, batchCells=100,
            nGenes=10000,
            batch_facLoc=0.1, batch_facScale=0.1,
            mean_rate=0.3, mean_shape=0.6,
            lib_loc=11, lib_scale=0.2, lib_norm=False,
            out_prob=0.05,
            out_facLoc=4, out_facScale=0.5,
            de_prob=0.1, de_downProb=0.1,
            de_facLoc=0.1, de_facScale=0.4,
            bcv_common=0.1, bcv_df=60,
            dropout_type='none', dropout_prob=0.5,
            dropout_mid=0, dropout_shape=-1,
            group_prob=1,
            path_from=0, path_length=100, path_skew=0.5,
            path_nonlinearProb=0.1, path_sigmaFac=0.8,
            seed=0
        """,
        body="""
            group_prob <- as.numeric(group_prob)
            path_from <- as.numeric(path_from)
            path_length <- as.numeric(path_length)
            path_skew <- as.numeric(path_skew)
            sim <- splatSimulate(
                method=method, batchCells=batchCells, nGenes=nGenes,
                batch.facLoc=batch_facLoc, batch.facScale=batch_facScale,
                mean.rate=mean_rate, mean.shape=mean_shape,
                lib.loc=lib_loc, lib.scale=lib_scale, lib.norm=lib_norm,
                out.prob=out_prob,
                out.facLoc=out_facLoc, out.facScale=out_facScale,
                de.prob=de_prob, de.downProb=de_downProb,
                de.facLoc=de_facLoc, de.facScale=de_facScale,
                bcv.common=bcv_common, bcv.df=bcv_df,
                dropout.type=dropout_type, dropout.mid=dropout_mid, dropout.shape=dropout_shape,
                group.prob=group_prob,
                path.from=path_from, path.length=path_length, path.skew=path_skew,
                path.nonlinearProb=path_nonlinearProb, path.sigmaFac=path_sigmaFac,
                seed=seed
            )
            data <- sim %>%
                counts() %>%
                t()
            list(data=data, time=sim$Step, branch=sim$Group)
        """)
    data = splat(method=method,
                 nBatches=nBatches, batchCells=batchCells,
                 nGenes=nGenes,
                 batch_facLoc=batch_facLoc, batch_facScale=batch_facScale,
                 mean_rate=mean_rate, mean_shape=mean_shape,
                 lib_loc=lib_loc, lib_scale=lib_scale, lib_norm=lib_norm,
                 out_prob=out_prob,
                 out_facLoc=out_facLoc, out_facScale=out_facScale,
                 de_prob=de_prob, de_downProb=de_downProb,
                 de_facLoc=de_facLoc, de_facScale=de_facScale,
                 bcv_common=bcv_common, bcv_df=bcv_df,
                 dropout_type=dropout_type, dropout_mid=dropout_mid,
                 dropout_shape=dropout_shape,
                 group_prob=group_prob,
                 path_from=path_from, path_length=path_length, path_skew=path_skew,
                 path_nonlinearProb=path_nonlinearProb, path_sigmaFac=path_sigmaFac,
                 seed=seed)
    if dropout_prob is not None:
        data['data'] = np.random.binomial(n=data['data'], p=1 - dropout_prob,
                                          size=data['data'].shape)
    return data


def _load_splat(dropout=0.5, sigma=0.18, method="paths", n_genes=17580,
                seed=None,
                data_name="Splatter Paths",
                **kwargs):
    np.random.seed(seed)
    print(kwargs)
    data = splatter(method=method, seed=seed,
                    batchCells=3000,  # 16825,
                    nGenes=17580,
                    mean_shape=6.6, mean_rate=0.45,
                    lib_loc=8.4 + np.log(2), lib_scale=0.33,
                    out_prob=0.016, out_facLoc=5.4, out_facScale=0.90,
                    bcv_common=sigma, bcv_df=21.6,
                    de_prob=0.2,
                    dropout_type="binomial", dropout_prob=dropout,
                    **kwargs,
                    )
    branch, labels = pd.factorize(data['branch']) if data[
        'branch'] is not None else (None, None)
    time = data['time']
    data = data['data']
    if n_genes < data.shape[1]:
        data = data[:, np.random.choice(data.shape[1], n_genes, replace=False)]
    data = scprep.normalize.library_size_normalize(data)
    data = scprep.transform.sqrt(data)
    data = PCA(data, n_components=100)
    return data, branch, True, labels, data_name, {}


def load_splat_paths(dropout=0.5, sigma=0.18, n_genes=17580, seed=42):
    np.random.seed(seed)
    n_groups = np.random.poisson(10)
    group_prob = np.random.dirichlet(np.ones(n_groups)).round(3)
    group_prob = group_prob / np.sum(group_prob)  # fix numerical error
    group_prob = group_prob.round(3)
    if np.sum(group_prob) != 1:
        group_prob[0] += 1 - np.sum(group_prob)
    group_prob = group_prob.round(3)
    path_nonlinearProb = np.random.uniform(0, 1)
    path_skew = np.random.beta(10, 10, n_groups)
    path_from = [0]
    for i in range(1, n_groups):
        path_from.append(np.random.randint(i))
    return _load_splat(dropout=dropout, sigma=sigma, n_genes=n_genes,
                       method="paths", group_prob=group_prob,
                       path_skew=path_skew,
                       path_from=path_from,
                       path_nonlinearProb=path_nonlinearProb,
                       data_name="Splatter Paths", seed=seed)


def load_splat_groups(dropout=0.5, sigma=0.18, n_genes=17580, seed=42):
    np.random.seed(seed)
    n_groups = np.random.poisson(10)
    group_prob = np.random.dirichlet(np.ones(n_groups)).round(3)
    group_prob = group_prob / np.sum(group_prob)  # fix numerical error
    group_prob = group_prob.round(3)
    if np.sum(group_prob) != 1:
        group_prob[0] += 1 - np.sum(group_prob)
    group_prob = group_prob.round(3)
    return _load_splat(dropout=dropout, sigma=sigma, n_genes=n_genes,
                       method="groups", group_prob=group_prob,
                       data_name="Splatter Groups", seed=seed)
