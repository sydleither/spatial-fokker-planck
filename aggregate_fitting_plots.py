from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from common import game_colors
from fokker_planck import param_names


def format_metric_name(metric):
    metric = metric.split(" ")
    new_metric_name = []
    for param_name in metric:
        if param_name == "awm":
            new_param_name = r'$\alpha_{wm}$'
        elif param_name == "amw":
            new_param_name = r'$\alpha_{mw}$'
        elif param_name == "sm":
            new_param_name = r'$s_m$'
        elif param_name == "mu":
            new_param_name = r'$\mu$'
        else:
            new_param_name = param_name
        new_metric_name.append(new_param_name)
    return " ".join(new_metric_name)


def plot_paramsweep_paper(save_loc, df, metrics, title):
    """
    Paper-ready plot of metrics across awm, amw, and sm.
    """
    sms = df["sm"].unique()
    fig, ax = plt.subplots(
        len(metrics), len(sms), figsize=(2 * len(sms), 2 * len(metrics)), constrained_layout=True
    )
    for j, metric in enumerate(metrics):
        min_distance = df[metric].min()
        max_distance = df[metric].max()
        if min_distance < 0:
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=min_distance, vmax=max_distance)
            cmap = plt.get_cmap("PuOr")
        else:
            norm = mcolors.Normalize(vmin=min_distance, vmax=max_distance)
            cmap = plt.get_cmap("Purples")
        scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i, sm in enumerate(sms):
            df_sm = df[df["sm"] == sm]
            df_sm = df_sm.pivot(index="amw", columns="awm", values=metric)
            ax[j][i].imshow(df_sm, cmap=cmap, norm=norm)
            ax[j][i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax[j][i].set_title(r'$s_m$='+str(sm))
        cbar = fig.colorbar(scalarmap, drawedges=False, ax=ax[j][-1])
        cbar.set_label(format_metric_name(metric))
    fig.suptitle(title)
    fig.supxlabel(r'$\alpha_{mw}$')
    fig.supylabel(r'$\alpha_{wm}$')
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{title}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_paramsweep(save_loc, df, metric):
    """
    Plot a metric across awm, amw, and sm.
    """
    sms = df["sm"].unique()
    fig, ax = plt.subplots(1, len(sms), figsize=(2 * len(sms), 2), constrained_layout=True)
    min_distance = df[metric].min()
    max_distance = df[metric].max()
    if min_distance < 0:
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=min_distance, vmax=max_distance)
        cmap = plt.get_cmap("PuOr")
    else:
        norm = mcolors.Normalize(vmin=min_distance, vmax=max_distance)
        cmap = plt.get_cmap("Purples")
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        awms = df_sm["awm"].unique()
        amws = df_sm["amw"].unique()
        df_sm = df_sm.pivot(index="amw", columns="awm", values=metric)
        ax[i].imshow(df_sm, cmap=cmap, norm=norm)
        ax[i].set_xticks(range(0, len(amws), 2), labels=amws[0::2])
        ax[i].set_yticks(range(0, len(awms), 2), labels=awms[0::2])
        ax[i].set_title(r'$s_m$='+str(sm))
    fig.supxlabel(r'$\alpha_{mw}$')
    fig.supylabel(r'$\alpha_{wm}$')
    cbar = fig.colorbar(scalarmap, drawedges=False, ax=ax[-1])
    cbar.set_label(format_metric_name(metric))
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{metric}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_paramsweep_game(save_loc, df):
    """
    Plot game quadrant across awm, amw, and sm.
    """
    sms = df["sm"].unique()
    fig, ax = plt.subplots(1, len(sms), figsize=(2 * len(sms), 2), constrained_layout=True)
    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        awms = df_sm["awm"].unique()
        amws = df_sm["amw"].unique()
        ax[i].scatter(df_sm["amw"], df_sm["awm"], c=df_sm["Game"], s=100)
        ax[i].set_xticks(range(0, len(amws), 2), labels=amws[0::2])
        ax[i].set_yticks(range(0, len(awms), 2), labels=awms[0::2])
        ax[i].set_title(r'$s_m$='+str(sm))
    fig.supxlabel(r'$\alpha_{mw}$')
    fig.supylabel(r'$\alpha_{wm}$')
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/game_quadrant.png", bbox_inches="tight", dpi=200)
    plt.close()


def get_confidence_interval_str(df, param_name):
    param_col = df[param_name]
    mean_diff = param_col.mean()
    sem_diff = param_col.sem()
    return f"{param_name}: {mean_diff} ({mean_diff-sem_diff}, {mean_diff+sem_diff})\n"


def plot_all(save_loc, df):
    metrics = [x for x in df.columns if x not in param_names]
    for metric in metrics:
        if metric == "Game":
            plot_paramsweep_game(save_loc, df)
        else:
            plot_paramsweep(save_loc, df, metric)
    game_params = [f"Mean {x} Difference" for x in ["sm", "awm", "amw"]]
    plot_paramsweep_paper(save_loc, df, game_params, "Self-Fitting Results")
    other_params = [f"Mean {x} Difference" for x in ["N", "mu", "c"]]
    plot_paramsweep_paper(save_loc, df, other_params, "Self-Fitting Results: N, mu, C")

    with open(f"{save_loc}/mean_param_diff_ci.txt", "w") as f:
        for param in param_names:
            param_name = f"Mean {param} Difference"
            f.write(get_confidence_interval_str(df, param_name))
        f.write("All Games\n")
        f.write("\t"+get_confidence_interval_str(df, "Correct Game Classifications"))
        df_unk = df[df["Game"] != game_colors["Unknown"]]
        f.write("Games Without Unknown\n")
        f.write("\t"+get_confidence_interval_str(df_unk, "Correct Game Classifications"))

    df.to_csv(f"{save_loc}/df.csv", index=False)
