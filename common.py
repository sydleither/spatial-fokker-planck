"""
Functions or variables used across multiple files.
"""

import os
from random import choices

import numpy as np
from scipy import stats


game_colors = {
    "Sensitive Wins": "#4C956C",
    "Coexistence": "#9C6D57",
    "Bistability": "#047495",
    "Resistant Wins": "#EF7C8E",
    "Unknown": "#929591",
}


def get_data_path(data_type:str, data_stage:str):
    """Get the path to the data if it exists, otherwise create the directories

    :param data_type: the name of the directory storing the data
    :type data_type: str
    :param data_stage: the stage of processing the data is at (raw, processed, etc)
    :type data_stage: str
    :return: the full path to the data
    :rtype: str
    """
    data_path = f"data/{data_type}/{data_stage}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def calculate_fp_params(a, b, c, d):
    """
    Convert from payoff terms to FP terms
    """
    epsilon = 1 - a
    awm = b + epsilon - 1
    sm = d + epsilon - 1
    amw = c + epsilon - 1 - sm
    return awm, amw, sm


def classify_game(awm, amw, sm, return_params=False):
    """
    Convert from FP terms to game quadrant
    """
    a = 1
    b = 1 + awm
    c = 1 + sm + amw
    d = 1 + sm

    if np.isclose(a, c) or np.isclose(b, d):
        game = "Unknown"
    elif a > c and b > d:
        game = "Sensitive Wins"
    elif c > a and b > d:
        game = "Coexistence"
    elif a > c and d > b:
        game = "Bistability"
    elif c > a and d > b:
        game = "Resistant Wins"

    if return_params:
        return game, a, b, c, d
    return game


def spatial_subsample(s_coords, r_coords, sample_length, num_samples=5000):
    """
    Create spatial subsample support and distribution.
    """
    dims = range(len(s_coords[0]))
    max_dims = [max(np.max(s_coords[:, i]), np.max(r_coords[:, i])) for i in dims]
    dim_vals = [choices(range(0, max_dims[i] - sample_length), k=num_samples) for i in dims]
    fr_counts = []
    for s in range(num_samples):
        ld = [dim_vals[i][s] for i in dims]
        ud = [ld[i] + sample_length for i in dims]
        subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
        subset_s = np.sum(np.all(subset_s, axis=0))
        subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
        subset_r = np.sum(np.all(subset_r, axis=0))
        subset_total = subset_s + subset_r
        if subset_total == 0:
            continue
        fr_counts.append(subset_r / subset_total)

    xdata = np.linspace(min(fr_counts), max(fr_counts), 100)
    kde = stats.gaussian_kde(fr_counts)
    pdf = kde(xdata)
    neg_lnpdf = -np.log(pdf)
    return xdata, neg_lnpdf
