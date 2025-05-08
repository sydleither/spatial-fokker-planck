"""
Use MCMC to find fokker-planck curves that fit mmmm params
"""

import sys

import numpy as np
import pandas as pd
import seaborn as sns

from common import get_data_path


def plot_mmmm(save_loc, file_name, df):
    """
    Plot the mmmm regime across awm, amw, sm.
    """
    facet = sns.FacetGrid(df, col="sm", hue="regime")
    facet.map_dataframe(sns.scatterplot, x="awm", y="amw", s=8)
    facet.add_legend()
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def define_regime(sm, sigma):
    if sm == sigma:
        return "maintenance"
    if sigma == 0 and sm != 0:
        return "masking"
    if sigma != 0 and sm == 0:
        return "mimicry"
    if sigma == -sm:
        return "mirroring"
    return "unk"


def mmmm(mu, amw, sm, sigma):
    awm_bot = (1+sigma) * (sm-2*mu+np.sqrt(4*mu**2+sm**2))
    if awm_bot == 0:
        print("Denominator of MMMM equation equals zero")
        print(f"\tmu={mu}, amw={amw}, sm={sm}, sigma={sigma}")
        return -np.inf
    awm_top = 2*sm*(sm-sigma) + (sm+2*mu-np.sqrt(4*mu**2+sm**2))*amw
    return awm_top / awm_bot


def calculate_sigma(mu, awm, amw, sm):
    sigma_bot = 2*sm + awm*(sm+np.sqrt(4*mu**2+sm**2)-np.sign(sm)*2*mu)
    if sigma_bot == 0:
        print("Denominator of sigma equation equals zero")
        print(f"\tmu={mu}, awm={awm}, amw={amw}, sm={sm}")
        return -np.inf
    sigma_top = 2*sm**2 + sm*(amw-awm) + (np.sign(sm)*2*mu-np.sqrt(4*mu**2+sm**2))*(amw+awm)
    return sigma_top / sigma_bot


def main(params):
    """
    Given mu, find mmmm regimes in the param space
    """
    mu = float(params[0])

    a_vals = np.round(np.arange(-0.5, 0.51, 0.02), 2)
    sm_vals = [-0.075, -0.05, -0.025, 0.025, 0.05, 0.075]
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                sigma = calculate_sigma(mu, awm, amw, sm)
                if sigma == -np.inf:
                    continue
                mmmm_awm = mmmm(mu, amw, sm, sigma)
                if mmmm_awm == -np.inf:
                    continue
                if np.isclose(awm, mmmm_awm):
                    regime = define_regime(sm, sigma)
                else:
                    regime = "none"
                data.append(
                    {
                        "awm": awm,
                        "amw": amw,
                        "sm": sm,
                        "regime": regime,
                    }
                )

    params_str = f"mmmm_mu={mu}"
    save_loc = get_data_path("self", ".")
    df = pd.DataFrame(data)
    plot_mmmm(save_loc, params_str, df)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide mu as an argument.")
    else:
        main(sys.argv[1:])
