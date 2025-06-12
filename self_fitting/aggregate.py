"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""

import sys

import numpy as np
import pandas as pd

from common import get_data_path
from fitting_utils import evaluate_performance
from fokker_planck import FokkerPlanck, param_names
from mcmc_utils import mcmc, plot_paramsweep


def main(params):
    """
    Iterate over amw, amw, and sm. Calculate quality of MCMC fit.
    """
    n = int(params[0])
    mu = float(params[1])

    fp = FokkerPlanck().fokker_planck_log
    xdata = np.linspace(0.01, 0.99, 100)

    a_vals = np.round(np.arange(-0.55, 0.551, 0.1), 2)
    sm_vals = np.round(np.arange(0.01, 0.1, 0.1), 2)
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                true_ydata = fp(xdata, n, mu, awm, amw, sm, 1)
                walker_ends = np.array(mcmc(fp, xdata, true_ydata))
                d = evaluate_performance(fp, xdata, true_ydata, walker_ends, n, mu, awm, amw, sm)
                data.append(d)

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    metrics = [x for x in data[0].keys() if x not in param_names]
    df = pd.DataFrame(data)
    for metric in metrics:
        plot_paramsweep(save_loc, df, metric)
    df.to_csv(f"{save_loc}/df.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
