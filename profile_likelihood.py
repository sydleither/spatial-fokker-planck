'''
Calculate the profile likelihood on one parameter set in each game quadrant
'''
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
    ranges = {"mu":np.linspace(0, 0.1, 100),
              "awm":np.linspace(-0.1, 0.1, 100),
              "amw":np.linspace(-0.1, 0.1, 100),
              "sm":np.linspace(-0.1, 0.1, 100)}
    param_idxs = {"mu":0, "awm":1, "amw":2, "sm":3}

    for game, param_set in parameters.items():
        profiles = {}
        popt, _ = curve_fit(fokker_planck_fixedn, xdata[game], ydata[game],
                            p0=[0, 0, 0, 0], bounds=(-0.5, 0.5))
        est_params = {k:popt[i] for i,k in enumerate(ranges)}
        for param_name in est_params:
            profiles[param_name] = []
            for new_param in ranges[param_name]:
                new_param_set = [v for k,v in est_params.items() if k != param_name]
                min_args = (xdata[game], ydata[game], param_idxs[param_name], new_param)
                result = minimize(fixed_residuals, x0=new_param_set, args=min_args)
                profiles[param_name].append(result.fun)

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


if __name__ == "__main__":
    main()
