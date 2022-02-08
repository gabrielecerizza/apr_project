import os

import pandas as pd
import torch
from torch import nn


def create_embeddings(
    model: nn.Module,
    base_path: str = "E:/Datasets/VoxCeleb1/subset/",
    num_secs: int = 6,
    feature_type: str = "logmel",
    strategy: str = "separate"
):
    embeddings_ls = []
    embeddings_dict = {
        "train": dict(),
        "val": dict(),
        "test": dict()
    }

    df = pd.read_csv(base_path + "subset_features.csv")
    df = df[df["Type"] == feature_type]
    for index, row in df.iterrows():
        # We compute embeddings only for
        # the original files
        if row["Augment"] != "none":
            continue

        file = row["File"]
        filename = os.path.splitext(os.path.basename(file))[0]
        features = torch.load(file).unsqueeze(1)
        embeddings = model(features)[0]

        if strategy == "mean":
            pass
        elif strategy == "separate":
            embeddings_file = base_path + "embeddings/" + row["Path"] \
                + filename + "_emb.pt"
            torch.save(embeddings, embeddings_file)

            embeddings_ls.append(
                row["Set"], 
                row["Speaker"], 
                row["Type"], 
                row["Augment"], 
                row["Seconds"], 
                row["Path"], 
                embeddings_file
            )
        else:
            raise ValueError("Invalid strategy argument")

    embeddings_df = pd.DataFrame(
        embeddings_ls, 
        columns = [
            "Set", "Speaker", "Type", "Augment", 
            "Seconds", "Path", "File"
        ]
    )
    embeddings_df.to_csv(base_path + "subset_embeddings.csv", index_label=False)


def compute_distances(batch):
    pass