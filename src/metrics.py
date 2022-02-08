import os
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_distances
from torch import nn
from tqdm.auto import tqdm


def create_embeddings(
    model: nn.Module,
    base_path: str = "E:/Datasets/VoxCeleb1/subset/",
    num_secs: int = 3,
    feature_type: str = "logmel",
    strategy: str = "separate"
):
    embeddings_ls = []

    df = pd.read_csv(base_path + f"subset_features_{num_secs}.csv")
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
                (
                    row["Set"], 
                    row["Speaker"], 
                    row["Type"], 
                    row["Augment"], 
                    row["Seconds"], 
                    row["Path"], 
                    embeddings_file
                )
            )
        else:
            raise ValueError("Invalid strategy argument")

    embeddings_df = pd.DataFrame(
        embeddings_ls, 
        columns=[
            "Set", "Speaker", "Type", "Augment", 
            "Seconds", "Path", "File"
        ]
    )
    embeddings_df.to_csv(
        base_path + f"subset_embeddings_{num_secs}.csv", 
        index_label=False
    )


def compute_scores(
    batch,
    base_path: str = "E:/Datasets/VoxCeleb1/subset/",
    num_secs: int = 3,
    top_n: int = 100
):
    """Compute scores and labels for the test/validation
    batch provided as argument. The scores are normalized
    according to the adaptive s-norm strategy, described
    in [1].

    Possibly a faster implementation here:
        https://github.com/juanmc2005/SpeakerEmbeddingLossComparison

    References
    ----------
        [1] P. Matějka et al., "Analysis of Score Normalization 
        in Multilingual Speaker Recognition," Proc. Interspeech 
        2017, pp. 1567-1571.
    """
    scores = []
    labels = []

    df = pd.read_csv(
        base_path + f"subset_embeddings_{num_secs}.csv"
    )

    speaker_embeddings = dict()

    for index, row in df.iterrows():
        if row["Set"] == "train":
            speaker = row["Speaker"]
            embedding_filename = row["File"]
            embedding = torch.load(embedding_filename)
            speaker_embeddings.setdefault(speaker, []).append(embedding) 

    speakers = list(speaker_embeddings.keys())
    cohort = np.vstack(
        [
            np.mean(
                np.vstack(speaker_embeddings[speaker]), 
                axis=0,
                keepdims=True
            ) 
            for speaker in speakers
        ]
    )

    for speaker, embedding in tqdm(
        speaker_embeddings.items(),
        desc="Computing scores",
        total=len(speaker_embeddings)
    ):
        e_distances = cosine_distances([embedding], cohort)[0]
        e_distances = np.sort(e_distances)[:top_n]

        me = np.mean(e_distances)
        se = np.std(e_distances)

        for idx, test_speaker in batch["speakers"]:
            test_embedding = batch["embeddings"][idx]

            distance = cosine_distances([embedding], [test_embedding])[0]

            t_distances = cosine_distances([test_embedding], cohort)[0]
            t_distances = np.sort(t_distances)[:top_n]

            mt = np.mean(t_distances)
            st = np.std(t_distances)

            e_term = (distance - me) / se
            t_term = (distance - mt) / st
            score = 0.5 * (e_term + t_term)

            scores.append(score)
            labels.append(int(speaker == test_speaker))

    return scores, labels


def eer(scores, labels):
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    assert np.isclose(eer1, eer2), (eer1, eer2)

    return eer1


def compute_error_rates(scores, labels):
    """Creates a list of false-negative rates, a list of 
    false-positive rates and a list of decision thresholds 
    that give those error-rates.

    Taken from the official VoxCeleb Speaker Recognition 
    Challenge 2021 implementation at:
        https://github.com/clovaai/voxceleb_trainer
    """

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


def min_dcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1):
    """Computes the minimum of the detection cost function.  
    The comments refer to equations in Section 3 of the NIST 2016 
    Speaker Recognition Evaluation Plan.

    Taken from the official VoxCeleb Speaker Recognition 
    Challenge 2021 implementation at:
        https://github.com/clovaai/voxceleb_trainer

    The default values for the parameters are taken from the
    NIST 2018 Speaker Recognition Evaluation Plan, AfV source
    type.
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
