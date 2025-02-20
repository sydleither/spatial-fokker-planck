'''
Functions or variables used across multiple files.
'''
import numpy as np


game_colors = {"sensitive_wins":"#4C956C", "coexistence":"#F97306",
               "bistability":"#047495", "resistant_wins":"#EF7C8E",
               "unknown":"gray"}
param_names = ["N", "mu", "awm", "amw", "sm"]


def get_sample_data(n, mu, include_fixed=False):
    '''
    One sample parameter set for each game quandrant
    '''
    parameters = {"sensitive_wins":{"awm":0.1, "amw":-0.1, "sm":0.05},
                  "coexistence":{"awm":0.06, "amw":-0.02, "sm":0.05},
                  "bistability":{"awm":0.025, "amw":-0.075, "sm":0.05},
                  "resistant_wins":{"awm":-0.05, "amw":0.05, "sm":0.05}}
    ydata = {}
    xdata = {}
    for game,param_set in parameters.items():
        if include_fixed:
            param_set["N"] = n
            param_set["mu"] = mu
        awm = param_set["awm"]
        amw = param_set["amw"]
        sm = param_set["sm"]
        p = np.linspace(0.01, 0.99, n)
        ydata[game] = fokker_planck(p, n, mu, awm, amw, sm)
        xdata[game] = p
    return parameters, xdata, ydata


def fokker_planck(x, n, mu, awm, amw, sm):
    '''
    The Fokker-Planck equation as defined in Barker-Clarke et al., 2024
    x = 0 or x = 1 will break the equation due to the log
    '''
    if awm == 0:
        fx = sm*x
    else:
        fx = (((1+sm)*awm+(1+awm)*amw)/awm**2)*np.log(1+awm*x) - ((awm+amw)/awm)*x
    phi = (1-2*n*mu)*np.log(x*(1-x)) - 2*n*fx - np.log(2*n)
    rho = np.exp(-phi)
    rho = rho / max(rho)
    if np.any(rho < 1e-31):
        return np.zeros(len(x))
    # rho = rho / (np.sum(rho)*x[0])
    # print(f"{rho} ({n}, {mu}, {awm}, {amw}, {sm})")
    # print(list((rho)[0::10]))
    # print()
    return rho


def calculate_fp_params(a, b, c, d):
    '''
    Convert from payoff terms to FP terms
    '''
    epsilon = 1 - a
    awm = b + epsilon - 1
    sm = d + epsilon - 1
    amw = c + epsilon - 1 - sm
    return awm, amw, sm


def classify_game(awm, amw, sm, return_params=False):
    '''
    Convert from FP terms to game quadrant
    '''
    a = 0
    b = awm
    c = sm+amw
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
