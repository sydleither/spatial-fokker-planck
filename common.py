"""
Functions or variables used across multiple files.
"""

import os

import numpy as np


game_colors = {
    "sensitive_wins": "#4C956C",
    "coexistence": "#9C6D57",
    "bistability": "#047495",
    "resistant_wins": "#EF7C8E",
    "unknown": "#929591",
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

    if a > c and b > d:
        game = "sensitive_wins"
    elif c > a and b > d:
        game = "coexistence"
    elif a > c and d > b:
        game = "bistability"
    elif c > a and d > b:
        game = "resistant_wins"
    else:
        game = "unknown"

    if return_params:
        return game, a, b, c, d
    return game


def get_regime_awms(mu, amw, sm, sigma):
    if sigma <= 0:
        print("sigma <= 0 invalid for getting regime awms")
        return
    regimes = {}
    regimes["maintenance"] = (mu*amw)/(sm*(1+sm))
    regimes["masking"] = amw + 2*sm
    regimes["mirroring"] = -((2*sm)/(1-sm)) + mu*((2*sm+amw)/(sm*(1-sm)))
    regimes["mimicry"] = -(sigma/(1+sigma)) + ((mu*(amw-sigma))/(sigma*(1+sigma)))
    return regimes


def get_regime_amws(mu, awm, sm, sigma):
    if sigma >= 0:
        print("sigma >= 0 invalid for getting regime amws")
        return
    regimes = {}
    regimes["maintenance"] = -((mu*awm*(1+sm))/sm)
    regimes["masking"] = awm - 2*sm
    regimes["mirroring"] = -2*sm - mu*((2*sm-awm*(1-sm))/sm)
    regimes["mimicry"] = sigma - (mu*(sigma+(1+sigma)*awm))/sigma
    return regimes


def calculate_sigma(mu, awm, amw, sm):
    sigma_bot = 2*sm + awm*(sm+np.sqrt(4*mu**2+sm**2)-np.sign(sm)*2*mu)
    if sigma_bot == 0:
        print("Denominator of sigma equation equals zero")
        print(f"\tmu={mu}, awm={awm}, amw={amw}, sm={sm}")
        return -np.inf
    sigma_top = 2*sm**2 + sm*(amw-awm) + (np.sign(sm)*2*mu-np.sqrt(4*mu**2+sm**2))*(amw+awm)
    return sigma_top / sigma_bot
