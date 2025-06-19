"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""

import sys

import numpy as np
import pandas as pd

from aggregate_fitting_plots import plot_all
from common import get_data_path
from fitting_utils import evaluate_performance
from fokker_planck import FokkerPlanck, param_names
from mcmc import mcmc


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
    df = pd.DataFrame(data)
    plot_all(save_loc, df)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
