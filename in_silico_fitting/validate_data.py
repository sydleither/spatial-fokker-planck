import json
import os
import sys

import pandas as pd
import seaborn as sns

from common import calculate_fp_params, classify_game, get_data_path


def main(data_type, source):
    data_path = get_data_path(f"{data_type}/{source}", "raw")
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
            data.append({
                "Time": time,
                "Proportion Resistant": resistant / (sensitive + resistant),
                "Game": game
            })

    df = pd.DataFrame(data)
    save_loc = get_data_path(f"{data_type}/{source}", "images")
    facet = sns.FacetGrid(
        df,
        col="Game",
        row="Time",
        height=4,
        aspect=1,
    )
    facet.map_dataframe(sns.kdeplot, x="Proportion Resistant", fill=True)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/fr_over_time.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
