import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import shutil, torch
from numpy.random import default_rng


def copy_audio(row, base_path):
    set_dir = row["Set"] + "/"
    audio_path = row["File"]

    src = base_path + "vox1_dev/" + audio_path
    if not os.path.isfile(src):
        src = base_path + "vox1_test/" + audio_path
    
    dst = base_path + "subset/raw/" + set_dir + audio_path
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


def create_subset(
    num_speakers: int = 20,
    base_path: str = "E:/Datasets/VoxCeleb1/"
):
    os.makedirs(base_path + "subset", exist_ok=True)
    
    # Get the official train, val, test splits
    ls = []
    with open(base_path + "iden_split.txt") as file:
        
        gender_df = pd.read_csv(base_path + "vox1_meta.csv", sep="\t")
        m_ratio = gender_df["Gender"].value_counts(normalize=True)["m"]
        f_ratio = gender_df["Gender"].value_counts(normalize=True)["f"]
        n_males = int(num_speakers * m_ratio)
        n_females = num_speakers - n_males
        male_ids = random.sample(
            list(
                gender_df[gender_df["Gender"] == "m"]["VoxCeleb1 ID"].unique()
            ),
            n_males
        )
        female_ids = random.sample(
            list(
                gender_df[gender_df["Gender"] == "f"]["VoxCeleb1 ID"].unique()
            ),
            n_females
        )
        chosen_ids = male_ids + female_ids

        for line in file:
            set_num, audio_path = line.split()
            speaker_id = audio_path.split("/")[0]
            if speaker_id not in chosen_ids:
                continue
            gender = list(
                gender_df[gender_df["VoxCeleb1 ID"] == speaker_id]["Gender"]
            )[0]
            ls.append((set_num, speaker_id, gender, audio_path))

    df = pd.DataFrame(ls, columns =["Set", "Speaker", "Gender", "File"])
    df["Set"] = df["Set"].apply(
        lambda x: "train" if x == "1" else "val" if x == "2" else "test"
    )
    
    print(df["Set"].value_counts())
    df.to_csv(base_path + "subset/subset.csv", index_label=False)

    m_sampled_ratio = df.drop_duplicates("Speaker")["Gender"].value_counts(normalize=True)["m"]
    f_sampled_ratio = df.drop_duplicates("Speaker")["Gender"].value_counts(normalize=True)["f"]

    print(
        f"Num speakers: {num_speakers}\n"
        f"Male ratio in dataset: {m_ratio}\n"
        f"Female ratio in dataset: {f_ratio}\n"
        f"Male sampled ratio: {m_sampled_ratio}\n"
        f"Female sampled ratio: {f_sampled_ratio}\n"
        f"Num sampled males: {n_males}\n"
        f"Num sampled females: {n_females}\n"
    )

    for index, row in df.iterrows():
        copy_audio(row, base_path)


def plot_waveform(
    waveform, 
    sample_rate, 
    title="Waveform", 
    xlim=None, 
    ylim=None
):
    """Plot waveform function from torchaudio tutorial."""

    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(
    spectrogram, 
    title=None, 
    ylabel="freq_bin", 
    aspect="auto", 
    xmax=None
):
    """Plot spectrogram function from torchaudio tutorial."""

    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(spectrogram, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def pad_tensor(x, x_len, max_len):
    pad_begin_len = torch.randint(
        low=0, high=max_len - x_len, size=(1,)
    )
    pad_end_len = max_len - x_len - pad_begin_len

    pad_begin = torch.zeros((x.shape[0], pad_begin_len))
    pad_end = torch.zeros((x.shape[0], pad_end_len))

    padded_tensor = torch.cat((pad_begin, x, pad_end), 1)
    return padded_tensor


def split_in_secs(waveform, sample_rate=16000, num_secs=3):
    waveform = waveform[0]
    num_samples = sample_rate * num_secs
    taken_samples = 0
    wav_ls = []

    while taken_samples < waveform.shape[0]:
        wav_ls.append(
            waveform[taken_samples:taken_samples + num_samples].unsqueeze(0)
        )
        taken_samples += num_samples

    # Pad last waveform if its length is less than
    # the desired number of seconds
    if wav_ls[-1].shape[1] < num_samples:
        wav_ls[-1] = pad_tensor(wav_ls[-1], wav_ls[-1].shape[1], num_samples)

    return wav_ls



    