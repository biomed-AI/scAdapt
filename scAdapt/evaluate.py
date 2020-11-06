import numpy as np
from universal_divergence import estimate
from sklearn.metrics import silhouette_samples, silhouette_score
import os
from math import log, e
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd

cal_min = 30  # minimum number of cells for estimation

# Reference: https://github.com/txWang/BERMUDA
def entropy(labels, base=None):
    """ Computes entropy of label distribution.
    Args:
        labels: list of integers
    Returns:
        ent: entropy
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


def cal_entropy(code, idx, dataset_labels, k=100):
    """ Calculate entropy of cell types of nearest neighbors
    Args:
        code: num_cells * num_features, embedding for calculating entropy
        idx: binary, index of observations to calculate entropy
        dataset_labels:
        k: number of nearest neighbors
    Returns:
        entropy_list: list of entropy of each cell
    """
    cell_sample = np.where(idx == True)[0]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(code)
    entropy_list = []
    _, indices = nbrs.kneighbors(code[cell_sample, :])
    for i in range(len(cell_sample)):
        entropy_list.append(entropy(dataset_labels[indices[i, :]]))

    return entropy_list


def batch_mixing_entropy(
        x: np.ndarray, y: np.ndarray, boots: int = 100,
        sample_size: int = 100, k: int = 100, metric: str = "minkowski",
        random_seed: int = 123, n_jobs: int = 1
):
    random_state = np.random.RandomState(random_seed)
    batches = np.unique(y)
    entropy = 0
    for _ in range(boots):
        bootsamples = random_state.choice(
            np.arange(x.shape[0]), sample_size, replace=False)
        subsample_x = x[bootsamples]
        neighbor = NearestNeighbors(
            n_neighbors=k, metric=metric, n_jobs=n_jobs
        )
        neighbor.fit(x)
        nn = neighbor.kneighbors(subsample_x, return_distance=False)
        for i in range(sample_size):
            for batch in batches:
                b = len(np.where(y[nn[i, :]] == batch)[0]) / k
                if b == 0:
                    entropy = entropy
                else:
                    entropy = entropy + b * np.log(b)
    entropy = -entropy / (boots * sample_size)
    return entropy

# Reference: https://github.com/gao-lab/Cell_BLAST/blob/fa1f30d2d54b68a06479513164746a80c1fdb031/Cell_BLAST/metrics.py
def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, k: float = 0.01, n: int = 1,
        metric: str = "minkowski", random_seed: int = 123,
        n_jobs: int = 1
):
    random_state = np.random.RandomState(random_seed)
    idx_list = [np.where(y == _y)[0] for _y in np.unique(y)]
    subsample_size = min(idx.size for idx in idx_list)
    subsample_scores = []
    for _ in range(n):
        subsample_idx_list = [
            random_state.choice(idx, subsample_size, replace=False)
            for idx in idx_list
        ]
        subsample_y = y[np.concatenate(subsample_idx_list)]
        subsample_x = x[np.concatenate(subsample_idx_list)]
        _k = subsample_y.shape[0] * k if k < 1 else k
        _k = np.round(_k).astype(np.int)
        nearestNeighbors = NearestNeighbors(
            n_neighbors=min(subsample_y.shape[0], _k + 1),
            metric=metric, n_jobs=n_jobs
        )
        nearestNeighbors.fit(subsample_x)
        nni = nearestNeighbors.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        subsample_scores.append(
            (_k - same_y_hits) * len(idx_list) /
            (_k * (len(idx_list) - 1))
        )
    return np.mean(subsample_scores)


def evaluate_scores(code_arr, cell_labels, dataset_labels, num_datasets, epoch):
    """ Calculate three proposed evaluation metrics
    Args:
        div_ent_code: num_cells * num_features, embedding for divergence and entropy calculation, usually with dim of 2
        sil_code: num_cells * num_features, embedding for silhouette score calculation
        cell_labels: true cell labels
        dataset_labels: index of different datasets
        num_datasets: number of datasets
    Returns:
        div_score: divergence score
        ent_score: entropy score
        sil_score: silhouette score
    """
    # calculate UMAP
    import umap
    fit = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, metric='cosine', random_state=123)
    div_ent_code = fit.fit_transform(code_arr)
    # div_ent_code = PCA(n_components=2).fit_transform(code_arr)
    # print(div_ent_code.shape)

    # calculate divergence and entropy
    div_pq = []  # divergence dataset p, q
    div_qp = []  # divergence dataset q, p
    div_pq_all = []  # divergence dataset p, q
    div_qp_all = []  # divergence dataset q, p
    ent = []  # entropy
    # pairs of datasets
    for d1 in range(1, num_datasets+1):
        for d2 in range(d1+1, num_datasets+1):
            idx1 = dataset_labels == d1
            idx2 = dataset_labels == d2 # the samples in dataset_labels belongs to which batch
            labels = np.intersect1d(np.unique(cell_labels[idx1]), np.unique(cell_labels[idx2])) #shared cluster between datasets
            idx1_mutual = np.logical_and(idx1, np.isin(cell_labels, labels))
            idx2_mutual = np.logical_and(idx2, np.isin(cell_labels, labels))
            idx_specific = np.logical_and(np.logical_or(idx1, idx2), np.logical_not(np.isin(cell_labels, labels)))

            # Estimate univesal k-NN divergence.
            if np.sum(idx1_mutual) >= cal_min and np.sum(idx2_mutual) >= cal_min:
                # calculate by cluster
                # batch_1 = div_ent_code[idx1, :]
                # batch_2 = div_ent_code[idx2, :]
                # for label_by in labels:
                #     # print(sum(label_by == cell_labels[idx1]), sum(label_by == cell_labels[idx2])) #cluster contain too little samples will lead to inf or nan
                #     #estimate(X, Y, k=None, n_jobs=1), X, Y: 2-dimensional array where each row is a sample.
                #     div_pq.append(
                #         estimate(batch_1[label_by == cell_labels[idx1], :], batch_2[label_by == cell_labels[idx2], :],
                #                  cal_min))
                #     div_qp.append(
                #         estimate(batch_2[label_by == cell_labels[idx2], :], batch_1[label_by == cell_labels[idx1], :],
                #                  cal_min))

                # calculate by all cells
                div_pq_all.append(max(estimate(div_ent_code[idx1_mutual, :], div_ent_code[idx2_mutual, :], cal_min), 0))
                div_qp_all.append(max(estimate(div_ent_code[idx2_mutual, :], div_ent_code[idx1_mutual, :], cal_min), 0))
            # entropy
            if (sum(idx_specific) > 0):
                ent_tmp = cal_entropy(div_ent_code, idx_specific, dataset_labels)
                ent.append(sum(ent_tmp) / len(ent_tmp))
    if len(ent) == 0:  # if no dataset specific cell types, store entropy as -1
        ent.append(-1)

    # # calculate silhouette_score
    # sil_code = code_arr
    # if sil_code.shape[1] > sil_dim:
    #     sil_code = PCA(n_components=2).fit_transform(sil_code)
    # sil_scores = silhouette_samples(sil_code, cell_labels, metric="euclidean")
    # print(div_ent_code.shape, sil_code.shape)

    sil_scores = silhouette_samples(div_ent_code, cell_labels, metric="euclidean")
    # sil_scores = silhouette_score(div_ent_code, cell_labels, metric="euclidean")

    # average for scores
    # div_pq = np.array(div_pq)[np.logical_and(np.isfinite(div_pq), ~np.isnan(div_pq))]
    # div_qp= np.array(div_qp)[np.logical_and(np.isfinite(div_qp), ~np.isnan(div_qp))]
    # div_score = (sum(div_pq) / len(div_pq) + sum(div_qp) / len(div_qp)) / 2
    div_score = 0
    div_score_all = (sum(div_pq_all) / len(div_pq_all) + sum(div_qp_all) / len(div_qp_all)) / 2
    ent_score = sum(ent) / len(ent)
    sil_score = sum(sil_scores) / len(sil_scores)

    alignment_score = seurat_alignment_score(code_arr, dataset_labels, n=10, k=0.01)
    mixing_entropy = batch_mixing_entropy(code_arr, dataset_labels)

    print("epoch: ", epoch, ' divergence_score: {:.3f}, {:.3f}, alignment_score, mixing_entropy: {:.3f},{:.3f} entropy_score: {:.3f}, silhouette_score: {:.3f}'.format(
        div_score,  div_score_all, alignment_score, mixing_entropy, ent_score, sil_score))

    return div_score, div_score_all, ent_score, sil_score

def evaluate_summary(code_arr, train_set, test_set, epoch):
    cell_labels = np.concatenate((train_set['labels'], test_set['labels']))

    train_size = train_set['features'].shape[0]
    test_size = test_set['features'].shape[0]
    total_cells = train_size + test_size
    dataset_labels = np.ones(total_cells, dtype=int)
    dataset_labels[train_size:total_cells] = 2
    num_datasets = 2

    div_score, div_score_all, ent_score, sil_score= evaluate_scores(code_arr, cell_labels, dataset_labels, num_datasets, epoch)
    return div_score, div_score_all, ent_score, sil_score

def evaluate_multibatch(code_arr, train_set, test_set, epoch):
    cell_labels = np.concatenate((train_set['labels'], test_set['labels']))

    train_size = train_set['features'].shape[0]
    test_size = test_set['features'].shape[0]
    total_cells = train_size + test_size
    dataset_labels = np.ones(total_cells, dtype=int)
    # dataset_labels[train_size:total_cells] = 2
    # num_datasets = 2

    num_datasets = 3 # there are three batches in the cross-species test: baron_mouse, TM_mouse, baron_human
    dataset_labels[1564:3429] = 2
    dataset_labels[3429:] = 3
    div_score, div_score_all, ent_score, sil_score = evaluate_scores(code_arr, cell_labels, dataset_labels, num_datasets, epoch)
    return div_score, div_score_all, ent_score, sil_score
