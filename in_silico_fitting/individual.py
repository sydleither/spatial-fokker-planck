"""
Run MCMC on spatial data
"""

import sys

from common import calculate_fp_params, get_data_path, spatial_subsample
from fokker_planck import FokkerPlanck
from in_silico_fitting.abm_utils import read_sample
from individual_fitting_plots import plot_all
from mcmc import mcmc


def main(data_type, source, sample):
    """
    Given data type, source, and sample
    Generate a spatial subsample distribution using the spatial data
    Fit Fokker-Planck to the distribution using MCMC
    """
    data_path = get_data_path(f"{data_type}/{source}", "raw")
    s_coords, r_coords, config = read_sample(data_path, sample)
    awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
    n = 100
    mu = 0
    c = 1

    fp = FokkerPlanck().fokker_planck_log
    xdata, ydata = spatial_subsample(s_coords, r_coords, 5)
    walker_ends = mcmc(fp, xdata, ydata, 100, 10000)

    save_loc = get_data_path(f"{data_type}/{source}", f"images/{sample}")
    plot_all(save_loc, fp, walker_ends, xdata, ydata, [n, mu, awm, amw, sm, c])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide the data type, source, and sample.")
    else:
        main(*sys.argv[1:])
