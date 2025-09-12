import argparse
import json
import os

import pandas as pd
import seaborn as sns

from common import calculate_fp_params, classify_game, game_colors, get_data_path


def plot_proportion_resistant(save_loc, df, games):
    facet = sns.FacetGrid(
        df,
        col="Game",
        col_order=games,
        row="Time",
        sharey=False,
        height=4,
        aspect=1,
    )
    facet.map_dataframe(sns.kdeplot, x="Proportion Resistant", fill=True)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/fr_over_time.png", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str, default="none")
    parser.add_argument("-d", "--data_type", type=str, default="in_silico")
    parser.add_argument("-src", "--source", type=str, default="5_5")
    parser.add_argument("-sub", "--subsample_length", type=int, default=5)
    parser.add_argument("-w", "--walkers", type=int, default=100)
    parser.add_argument("-i", "--iterations", type=int, default=50000)
    args = parser.parse_args()

    data_path = get_data_path(f"{args.data_type}/{args.source}", "raw")
    data = []
    for sample in os.listdir(data_path):
        if os.path.isfile(f"{data_path}/{sample}"):
            continue
        config = json.loads(open(f"{data_path}/{sample}/{sample}.json").read())
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        game = classify_game(awm, amw, sm)
        df = pd.read_csv(f"{data_path}/{sample}/{sample}/2Dcoords.csv")
        df = df[df["time"] > 0]
        for time in df["time"].unique():
            df_time = df[df["time"] == time]
            sensitive = len(df_time[df_time["type"] == 0])
            resistant = len(df_time[df_time["type"] == 1])
            data.append(
                {
                    "Sample": sample,
                    "Time": time,
                    "Proportion Resistant": resistant / (sensitive + resistant),
                    "Game": game,
                }
            )

    df = pd.DataFrame(data)
    save_loc = get_data_path(f"{args.data_type}/{args.source}", "images")
    games = {g: c for g, c in game_colors.items() if g != "Unknown"}
    plot_proportion_resistant(save_loc, df, games)


if __name__ == "__main__":
    main()
