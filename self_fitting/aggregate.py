"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""

import sys

import numpy as np
import pandas as pd

from common import get_data_path
from fitting_utils import evaluate_performance
from fokker_planck import FokkerPlanck, param_names
from mcmc_utils import mcmc, plot_paramsweep, plot_paramsweep_game


def main(params):
    """
    Iterate over amw, amw, and sm. Calculate quality of MCMC fit.
    """
    n = int(params[0])
    mu = float(params[1])
    c = 1

    fp = FokkerPlanck().fokker_planck_log
    xdata = np.linspace(0.01, 0.99, 100)
    scale = 0.5

    data = []
    sm_vals = np.round(np.arange(0.05, 0.55, 0.1), 2)
    for sm in sm_vals:
        for awm in np.round(np.arange(-2*sm, 4*sm+0.1*sm*scale, sm*scale), 3):
            for amw in np.round(np.arange(-4*sm, 2*sm+0.1*sm*scale, sm*scale), 3):
                true_ydata = fp(xdata, n, mu, awm, amw, sm, c)
                walker_ends = np.array(mcmc(fp, xdata, true_ydata))
                d = evaluate_performance(fp, xdata, true_ydata, walker_ends, n, mu, awm, amw, sm, c)
                data.append(d)

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    metrics = [x for x in data[0].keys() if x not in param_names]
    df = pd.DataFrame(data)
    for metric in metrics:
        if metric == "game":
            plot_paramsweep_game(save_loc, df)
        else:
            plot_paramsweep(save_loc, df, metric)

    with open(f"{save_loc}/mean_param_diff_ci.txt", "w") as f:
        for param in param_names:
            param_col = df[f"mean_{param}_diff"]
            mean_diff = param_col.mean()
            sem_diff = param_col.sem()
            f.write(f"{param}: {mean_diff} ({mean_diff-sem_diff}, {mean_diff+sem_diff})\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
