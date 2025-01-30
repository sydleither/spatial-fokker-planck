'''
Calculate the profile likelihood on one parameter set in each game quadrant
'''
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

from common import fokker_planck_fixedn, get_sample_data


def fixed_residuals(param_set, xdata, ydata, fixed_param_idx, fixed_param_val):
    '''
    Get sum of squares error between true params and estimated params
    Assumes given param_set does not include fixed param
    '''
    param_set = np.insert(param_set, fixed_param_idx, fixed_param_val)
    return np.sum((ydata - fokker_planck_fixedn(xdata, *param_set))**2)


def main():
    '''
    Profile likelihood
    '''
    parameters, xdata, ydata = get_sample_data()
    ranges = {"mu":np.linspace(0, 0.1, 101),
              "awm":np.linspace(-0.1, 0.1, 101),
              "amw":np.linspace(-0.1, 0.1, 101),
              "sm":np.linspace(-0.1, 0.1, 101)}
    param_idxs = {"mu":0, "awm":1, "amw":2, "sm":3}
    bounds = (-1, 1)

    for game, param_set in parameters.items():
        profiles = {}
        res_params = {}
        popt, _ = curve_fit(fokker_planck_fixedn, xdata[game], ydata[game],
                            p0=[0, 0, 0, 0], bounds=bounds)
        est_params = {k:popt[i] for i,k in enumerate(ranges)}
        for param_name in est_params:
            profiles[param_name] = []
            res_params[param_name] = []
            for i,new_param in enumerate(ranges[param_name]):
                new_param_set = [v for k,v in est_params.items() if k != param_name]
                min_args = (xdata[game], ydata[game], param_idxs[param_name], new_param)
                result = minimize(fixed_residuals, x0=new_param_set,
                                  bounds=(bounds, bounds, bounds), args=min_args)
                profiles[param_name].append(result.fun)
                if i % 10 == 0:
                    res_param_set = [str(x) for x in result.x]
                    res_param_set.insert(param_idxs[param_name], str(new_param))
                    res_params[param_name].append(" ".join(res_param_set))

        fig, ax = plt.subplots(1, len(ranges), figsize=(5*len(ranges), 5))
        for i,param_name in enumerate(profiles):
            ax[i].scatter(ranges[param_name], profiles[param_name], color="black")
            ax[i].axvline(param_set[param_name], color="hotpink")
            ax[i].axvline(est_params[param_name], color="green")
            ax[i].set(title=param_name)
        fig.supxlabel("Parameter")
        fig.supylabel("Residual")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        fig.savefig(f"profile_{game}.png")
        plt.close()

        json_string = json.dumps(res_params, indent=4)
        with open(f"profile_{game}.txt", "w") as text_file:
            text_file.write(json_string)


if __name__ == "__main__":
    main()
