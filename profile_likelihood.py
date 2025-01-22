'''
Calculate the profile likelihood on one parameter set in each game quadrant
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

from common import fokker_planck_fixedn, get_sample_data, residuals


def main():
    parameters, xdata, ydata = get_sample_data()
    ranges = {"mu":np.linspace(0, 0.1, 100),
              "awm":np.linspace(-0.1, 0.1, 100),
              "amw":np.linspace(-0.1, 0.1, 100),
              "sm":np.linspace(-0.1, 0.1, 100)}

    for game, param_set in parameters.items():
        profiles = {}
        popt, _ = curve_fit(fokker_planck_fixedn, xdata[game], ydata[game],
                            p0=[0, 0, 0, 0], bounds=(-0.5, 0.5))
        est_params = {k:popt[i] for i,k in enumerate(ranges)}
        for param_name in est_params:
            profiles[param_name] = []
            for new_param in ranges[param_name]:
                new_param_set = [new_param if k == param_name else v for k,v in est_params.items()]
                result = minimize(residuals, x0=new_param_set, args=(xdata[game], ydata[game]))
                profiles[param_name].append(result.fun)

        fig, ax = plt.subplots(1, len(ranges), figsize=(5*len(ranges), 5))
        for i,param_name in enumerate(profiles):
            ax[i].scatter(ranges[param_name], profiles[param_name], color="black")
            ax[i].axvline(param_set[param_name], color="pink")
            ax[i].axvline(est_params[param_name], color="green")
            ax[i].set(title=param_name)
        fig.supxlabel("Parameter")
        fig.supylabel("Residual")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        fig.savefig(f"profile_{game}.png")


if __name__ == "__main__":
    main()
