"""
Use MCMC to find fokker-planck curves that fit mmmm params
"""

import sys

import numpy as np
import pandas as pd

from common import get_data_path
from mcmc_utils import plot_paramsweep


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


def main(params):
    """
    Given mu, find mmmm regimes in the param space
    """
    mu = float(params[0])

    a_vals = np.round(np.arange(-0.5, 0.51, 0.05), 2)
    sm_vals = [0.025, 0.05, 0.075]
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                sigma = calculate_sigma(mu, awm, amw, sm)
                if sigma == -np.inf:
                    continue
                if sigma > 0:
                    awm_regimes = get_regime_awms(mu, amw, sm, sigma)
                    distances = {k:v-awm for k,v in awm_regimes.items()}
                elif sigma < 0:
                    amw_regimes = get_regime_amws(mu, awm, sm, sigma)
                    distances = {k:v-amw for k,v in amw_regimes.items()}
                else:
                    continue
                params = {"awm": awm, "amw": amw, "sm": sm, "sigma":sigma}
                data.append(params | distances)

    params_str = f"mu={mu}"
    save_loc = get_data_path("self", params_str)
    df = pd.DataFrame(data)
    plot_paramsweep(save_loc, df, "maintenance")
    plot_paramsweep(save_loc, df, "masking")
    plot_paramsweep(save_loc, df, "mirroring")
    plot_paramsweep(save_loc, df, "mimicry")
    plot_paramsweep(save_loc, df, "sigma")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide mu as an argument.")
    else:
        main(sys.argv[1:])
