import json

import pandas as pd


def read_sample(data_path, sample, time=None):
    df = pd.read_csv(f"{data_path}/{sample}/{sample}/2Dcoords.csv")
    if time is None:
        df = df[df["time"] == df["time"].max()]
    else:
        df = df[df["time"] == time]
    s_coords = df.loc[df["type"] == 0][["x", "y"]].values
    r_coords = df.loc[df["type"] == 1][["x", "y"]].values
    config = json.loads(open(f"{data_path}/{sample}/{sample}.json").read())
    return s_coords, r_coords, config
