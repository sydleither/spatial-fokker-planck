'''
Visualize the curve from the parameter sets defined in common
'''
import matplotlib.pyplot as plt

from common import classify_game, game_colors, get_sample_data


def visualize_curves(params, xdata, ydata,):
    '''
    Visualize the FP solutions, each curve on a different plot
    '''
    num_curves = len(params)
    fig, ax = plt.subplots(1, num_curves, figsize=(5*num_curves, 5))
    if num_curves == 1:
        ax = [ax]
    for i,game in enumerate(params):
        param_set = params[game]
        fp_sol = ydata[game]
        classified_game = classify_game(param_set["awm"], param_set["amw"], param_set["sm"])
        ax[i].plot(xdata[game], fp_sol/max(fp_sol), color=game_colors[classified_game], linewidth=3)
        ax[i].set(xlim=(0,1), ylim=(0,1))
        ax[i].set(title=" ".join([f"{k}={v}" for k,v in param_set.items()]))
    fig.supxlabel("Fraction Mutant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig("fp.png", bbox_inches="tight")


def main():
    parameters, xdata, ydata = get_sample_data()
    visualize_curves(parameters, xdata, ydata)


if __name__ == "__main__":
    main()
