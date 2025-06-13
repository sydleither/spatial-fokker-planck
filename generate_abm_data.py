"""Generate data used for main experiments.

Writes config files and run scripts to sample the ABM
across diverse payoff matrices and starting conditions.

Expected usage:
python3 -m data_generation.main_data data_dir interaction_radius reproduction_radius run_cmd

Where:
data_dir: the parent directory the data will be located in
interaction_radius
reproduction_radius
run_cmd: how to run the ABM samples
    e.g. "sbatch job_abm.sb" or "java -cp build/:lib/* SpatialEGT.SpatialEGT"
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import classify_game, get_data_path
from EGT_HAL.config_utils import latin_hybercube_sample, write_config, write_run_scripts
from mcmc_utils import gamespace_plot


def game_spread(data_dir, experiment_name, samples):
    df = pd.DataFrame(samples, columns=["awm", "amw", "sm"])
    df[["game", "a", "b", "c", "d"]] = df.apply(
        lambda x: classify_game(x["awm"], x["amw"], x["sm"], True), axis=1, result_type="expand"
    )
    df["c-a"] = df["c"] - df["a"]
    df["b-d"] = df["b"] - df["d"]

    with open(f"{data_dir}/{experiment_name}/gamespread.txt", "w") as f:
        f.write(df.groupby("game").count().to_string())

    fig, ax = plt.subplots(figsize=(4, 4))
    gamespace_plot(ax, df, "c-a", "b-d")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{data_dir}/{experiment_name}/gamespace.png", bbox_inches="tight")
    plt.close()


def main(data_dir, interaction_radius, reproduction_radius, run_command):
    """
    Generate scripts to run the ABM
    """
    data_dir = get_data_path(data_dir, "raw")
    experiment_name = f"{interaction_radius}_{reproduction_radius}"
    space = "2D"
    end_time = 100
    grid_size = 200
    scale = 0.5

    samples = []
    for sm in np.round(np.arange(0.05, 0.55, 0.1), 2):
        for awm in np.round(np.arange(-2*sm, 4*sm+0.1*sm*scale, sm*scale), 3):
            for amw in np.round(np.arange(-4*sm, 2*sm+0.1*sm*scale, sm*scale), 3):
                samples.append(({"awm": awm, "amw": amw, "sm":sm}))

    run_output = []
    run_str = f"{run_command} ../{data_dir} {experiment_name}"
    for s, sample in enumerate(samples):
        config_name = str(s)
        seed = config_name
        game, a, b, c, d = classify_game(sample["awm"], sample["amw"], sample["sm"], return_params=True)
        if game == "unknown":
            continue
        payoff = [a, b, c, d]
        write_config(
            data_dir,
            experiment_name,
            config_name,
            seed,
            payoff,
            int(0.5 * grid_size**2),
            0.5,
            x=grid_size,
            y=grid_size,
            interaction_radius=interaction_radius,
            reproduction_radius=reproduction_radius,
            write_freq=end_time,
            ticks=end_time,
        )
        run_output.append(f"{run_str} {config_name} {space} {seed}\n")
    write_run_scripts(data_dir, experiment_name, run_output)
    game_spread(data_dir, experiment_name, samples)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        print("Please see the module docstring for usage instructions.")
