'''
Calculate the error in estimating input params
'''
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from common import classify_game, fokker_planck, game_colors, param_names


n = 0
mu = 0
def fokker_planck_fixed(x, awm, amw, s):
    '''
    The Fokker-Planck equation with a fixed N and mu
    '''
    return fokker_planck(x, n, mu, awm, amw, s)


global_param_history = []
def residuals_callback(param_set, xdata, ydata):
    '''
    Get error between true params and estimated params, to plug into scipy least_squares
    Also tracks the params least_squares tries as it optimizes
    '''
    global_param_history.append(param_set.copy())
    return fokker_planck_fixed(xdata, *param_set) - ydata


def format_vals(vals):
    '''
    Take in list of params and return formatted strings
    '''
    return [f"{x:6.3f}" for x in vals]


def main(params):
    '''
    Given a set of params (N, mu, awm, amw, sm)
    Generate a distribution using Fokker-Planck equation with those params
    Give the distribution and Fokker-Planck equation to a solver
    Compare solver-estimated params to true params
    Plot resulting curves and solver search process
    '''
    global global_param_history, n, mu
    params = [float(x) for x in params]
    n = int(params[0])
    mu = params[1]
    true_params = params[2:]
    xdata = np.linspace(0.01, 0.99, n)
    ydata = fokker_planck(xdata, n, mu, *true_params)

    result = least_squares(residuals_callback, (0, 0, 0), args=(xdata, ydata), bounds=(-0.5, 0.5))
    est_params = result.x

    estimated = fokker_planck_fixed(xdata, *est_params)
    param_histories = list(zip(*global_param_history.copy()))

    param_colors = ["#75bbfd", "#bf77f6", "#f7879a"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(xdata, ydata/max(ydata), lw=2, label="True",
               color=game_colors[classify_game(*true_params)])
    ax[0].plot(xdata, estimated/max(estimated), lw=3, label="Estimated",
               color=game_colors[classify_game(*est_params)], ls="--")
    ax[0].set(xlabel="Fraction Mutant", ylabel="Probability Density")
    ax[0].legend()
    est_param_names = param_names[2:]
    for i,param_history in enumerate(param_histories):
        ax[1].plot(range(len(param_history)), param_history,
                   color=param_colors[i], label=est_param_names[i])
        ax[1].axhline(true_params[i], color=param_colors[i], linestyle="dashed")
        ax[1].set(xlabel="Iteration", ylabel="Parameter Value")
        ax[1].legend()
    fig.suptitle(", ".join(format_vals(true_params))+"\n"+", ".join(format_vals(est_params)))
    fig.tight_layout()
    fig.patch.set_alpha(0)
    file_name = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    fig.savefig(f"fit_{file_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
