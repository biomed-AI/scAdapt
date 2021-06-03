# !/usr/bin/env python
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

# The 954 most common RGB monitor colors
# https://xkcd.com/color/rgb/
# 9-class Set1, for plotting data with qualitative labels
color_dict = {0:'#e41a1c', 1:'#377eb8', 2:'#4daf4a', 3:'#984ea3', 4:'#ff7f00',
              5:'#ffff33', 6:'#a65628', 7:'#f781bf', 8:'#999999', 9:'#00ffff', 10: '#96f97b'}

import torch
def matrix_one_hot(x, class_count):
	return torch.eye(class_count)[x,:]

def variable_to_numpy(x):
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        return float(np.sum(ans))
    return ans

def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix

def cal_UMAP(code, pca_dim = 50, n_neighbors = 30, min_dist=0.1, n_components=2, metric='cosine'):
    """ Calculate UMAP dimensionality reduction
    Args:
        code: num_cells * num_features
        pca_dim: if dimensionality of code > pca_dim, apply PCA first
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        n_components: UMAP parameter
        metric: UMAP parameter
    Returns:
        umap_code: num_cells * n_components
    """
    if code.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        code = pca.fit_transform(code)
    fit = umap.UMAP(n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=n_components,
                    metric=metric,
                    random_state=123)
    umap_code = fit.fit_transform(code)

    return umap_code


def plot_labels(coor_code, labels, label_dict, axis_name, save_path, method_name):
    ''' Plot cells with qualitative labels
    Args:
        coor_code: num_cells * 2 matrix for visualization
        labels: labels in integer
        label_dict: dictionary converting integer to labels names
        axis_name: list of two, names of x and y axis
        save_path: path to save the plot
    Returns:
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = np.unique(labels)
    unique_labels.sort()
    for i in range(len(unique_labels)):
        g = unique_labels[i]
        ix = np.where(labels == g)
        ax.scatter(coor_code[ix, 0], coor_code[ix, 1], s=1, c=color_dict[i % len(color_dict)], label=label_dict[g], alpha=0.7)
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_title(method_name, fontsize = 18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.14,1.0),ncol=1,
                prop={'size': 6}, markerscale=5.0,frameon=False)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_expr(coor_code, vals, axis_name, save_path):
    ''' Plot cells with continuous expression levels
    Args:
        coor_code: num_cells * 2 matrix for visualization
        vals: expression values
        axis_name: list of two, names of x and y axis
        save_path: path to save the plot
    Returns:
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    # random permutate to solve covering issue of datasets in the visualization
    tmp = np.argsort(vals)
    coor_code = coor_code[tmp,:]
    vals = vals[tmp]
    g = ax.scatter(coor_code[:, 0], coor_code[:, 1], s=1, c=vals, cmap='viridis',alpha=0.2)
    g.set_facecolor('none')
    fig.colorbar(g)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, save_path):
    ''' Plot loss versus epochs
    Args:
        loss_total_list: list of total loss
        loss_reconstruct_list: list of reconstruction loss
        loss_transfer_list: list of transfer loss
        save_path: path to save the plot
    Returns:
    '''
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(range(len(loss_total_list)), loss_total_list, "r:",linewidth=1)
    ax[0].legend(['total_loss'])
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].plot(range(len(loss_reconstruct_list)), loss_reconstruct_list, "b--",linewidth=1)
    ax[1].plot(range(len(loss_transfer_list)), loss_transfer_list, "g-",linewidth=1)
    ax[1].legend(['loss_reconstruct', 'loss_transfer'])
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
