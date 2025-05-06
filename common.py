"""
Functions or variables used across multiple files.
"""

import os


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
    a = 0
    b = awm
    c = sm + amw
    d = sm

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
