"""Generate data used for main experiments.

Writes config files and run scripts to sample the ABM
across diverse payoff matrices and starting conditions.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import classify_game, get_data_path
from EGT_HAL.config_utils import write_config, write_run_scripts
from fitting_utils import game_parameter_sweep
from individual_fitting_plots import gamespace_plot


def game_spread(data_dir, experiment_name, samples):
    df = pd.DataFrame(samples, columns=["awm", "amw", "sm"])
    df[["Game", "a", "b", "c", "d"]] = df.apply(
        lambda x: classify_game(x["awm"], x["amw"], x["sm"], return_params=True),
        axis=1,
        result_type="expand",
    )
    df["c-a"] = df["c"] - df["a"]
    df["b-d"] = df["b"] - df["d"]

    with open(f"{data_dir}/{experiment_name}/gamespread.txt", "w") as f:
        f.write(df.groupby("Game").count().to_string())

    fig, ax = plt.subplots(figsize=(4, 4))
    gamespace_plot(ax, df, "c-a", "b-d")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{data_dir}/{experiment_name}/gamespace.png", bbox_inches="tight", dpi=200)
    plt.close()


def main():
    """
    Generate scripts to run the ABM
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="in_silico")
    parser.add_argument("-i", "--interaction_radius", type=int, default=5)
    parser.add_argument("-r", "--reproduction_radius", type=int, default=5)
    parser.add_argument("-run", "--run_command", type=str, default="sbatch job_abm.sb")
    parser.add_argument("-exp", "--experiment_name", type=str, default="raw")
    parser.add_argument("-end", "--end_time", type=int, default=100)
    parser.add_argument("-w", "--write_freq", type=int, default=100)
    parser.add_argument("-g", "--grid_size", type=int, default=200)
    parser.add_argument("-mu", "--mutation_rate", type=float, default=0.01)
    args = parser.parse_args()

    source = f"{args.interaction_radius}_{args.reproduction_radius}"
    data_dir = get_data_path(args.data_dir, source)

    game_parameters = game_parameter_sweep()
    samples = []
    for awm, amw, sm in game_parameters:
        samples.append(({"awm": awm, "amw": amw, "sm": sm}))

    run_output = []
    run_str = f"{args.run_command} ../{data_dir} {args.experiment_name}"
    for s, sample in enumerate(samples):
        config_name = str(s)
        seed = config_name
        _, a, b, c, d = classify_game(
            sample["awm"], sample["amw"], sample["sm"], norm=0.5, return_params=True
        )
        payoff = [a, b, c, d]
        write_config(
            data_dir,
            args.experiment_name,
            config_name,
            seed,
            payoff,
            int(0.5 * args.grid_size**2),
            0.5,
            x=args.grid_size,
            y=args.grid_size,
            turnover=0.125,
            mutation_rate=args.mutation_rate,
            interaction_radius=args.interaction_radius,
            reproduction_radius=args.reproduction_radius,
            write_freq=args.write_freq,
            ticks=args.end_time,
        )
        run_output.append(f"{run_str} {config_name} 2D {seed}\n")
    write_run_scripts(data_dir, args.experiment_name, run_output)
    game_spread(data_dir, args.experiment_name, samples)


if __name__ == "__main__":
    main()
