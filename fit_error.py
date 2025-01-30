'''
Calculate the error in estimating parameters on one parameter set in each game quadrant
'''
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from common import fokker_planck_fixedn, game_colors, get_sample_data


global_param_history = []
def residuals_callback(param_set, xdata, ydata):
    '''
    Get error between true params and estimated params, to plug into scipy least_squares
    Also tracks the parameters least_squares tries as it optimizes
    '''
    global_param_history.append(param_set.copy())
    return fokker_planck_fixedn(xdata, *param_set) - ydata


def main():
    '''
    Given a set of parameters (N, mu, awm, amw, sm)
    Generate a distribution using Fokker-Planck equation with those parameters
    Give the distribution and Fokker-Planck equation to a solver
    Compare solver-estimated parameters to true parameters
    Plot resulting curves and solver search process
    '''
    global global_param_history
    parameters, xdata, ydata = get_sample_data()
    initial_guess = (0, 0, 0, 0)
    bounds = (-0.5, 0.5)

    local_param_history = {}
    errors = {}
    estimated = {}
    for game,true_params in parameters.items():
        result = least_squares(residuals_callback, initial_guess,
                               args=(xdata[game], ydata[game]), bounds=bounds)
        popt = result.x
        estimated[game] = fokker_planck_fixedn(xdata[game], *popt)
        errors[game] = {k:abs(true_params[k]-popt[i]) for i,k in enumerate(true_params)}
        local_param_history[game] = list(zip(*global_param_history.copy()))
        global_param_history = []
        print(game)
        print(f"\tTrue Parameters: {true_params}")
        print(f"\tEst Parameters: {popt}")

    fig, ax = plt.subplots(2, len(parameters), figsize=(5*len(parameters), 10))
    for i,game in enumerate(parameters):
        ax[0,i].plot(xdata[game], ydata[game]/max(ydata[game]),
                     linewidth=2, color=game_colors[game])
        ax[0,i].plot(xdata[game], estimated[game]/max(estimated[game]),
                     linestyle="dashed", linewidth=2, color="dimgray")
        ax[0,i].set(title=game, xlabel="Fraction Mutant", ylabel="Probability Density")
        ax[1,i].bar(parameters[game].keys(), errors[game].values())
        ax[1,i].set(xlabel="Parameter", ylabel="Error")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig("fit_error.png", bbox_inches="tight")

    colors = ["blue", "orange", "green", "red"]
    fig, ax = plt.subplots(1, len(parameters), figsize=(5*len(parameters), 5))
    for i,game in enumerate(parameters):
        param_names = list(parameters[game].keys())
        for j,game_param_history in enumerate(local_param_history[game]):
            ax[i].scatter(range(len(game_param_history)), game_param_history,
                          color=colors[j], label=param_names[j])
            ax[i].axhline(parameters[game][param_names[j]], color=colors[j])
            ax[i].set(title=game, xlabel="Iteration", ylabel="Parameter Value")
            ax[i].legend()
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig("fit_error_param_history.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
