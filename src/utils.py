import math
import os
import pathlib
import random
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchaudio
import torchaudio.transforms as T
from pedalboard import Pedalboard, Reverb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


class RandomClip:
    def __init__(
        self, 
        sample_rate: int = 16000,
        clip_secs: int = 3
    ):
        self.clip_length = clip_secs * sample_rate

    def __call__(self, audio_data):
        audio_data = audio_data
        audio_length = audio_data.shape[-1]
        if audio_length > self.clip_length:
            audio_data = audio_data[0]
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[offset:(offset+self.clip_length)]
            audio_data = audio_data.unsqueeze(0)
        elif audio_length < self.clip_length:
            audio_data = audio_data.unsqueeze(0)
            audio_data = pad_tensor(
                audio_data, audio_length, self.clip_length
            )[0]

        return audio_data


class RandomSpeedChange:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        audio_data = audio_data[0].numpy()
        speed_factor = random.choice([0.9, 1.1])

        transformed_audio = librosa.effects.time_stretch(
            y=audio_data, rate=speed_factor
        )
        return torch.tensor(np.array([transformed_audio]))


class RandomBackgroundNoise:
    """Adds a random noise to the waveform. Noises
    are taken from the Musan dataset [1] by default. 
    The signal-to-noise ratio DB controls the volume 
    of the noise.

    The code was adapted from:
        https://jonathanbgn.com/2021/08/30/audio-augmentation.html

    References
    ----------
        [1] D. Snyder, G. Chen and D. Povey, "MUSAN: A Music, 
        Speech, and Noise Corpus", 2015, https://arxiv.org/abs/1510.08484.
    """
    def __init__(
        self, 
        sample_rate: int = 16000, 
        noise_dir: str = "E:/Datasets/Musan/noise",
        min_snr_db: int = 0, 
        max_snr_db: int = 15
    ):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob("**/*.wav"))
        if len(self.noise_files_list) == 0:
            raise IOError(
                f"No .wav file found in the noise directory '{noise_dir}'"
            )

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        noise, noise_sr = torchaudio.load(random_noise_file)
        noise = librosa.resample(
            y=noise[0].numpy(), 
            orig_sr=noise_sr, 
            target_sr=self.sample_rate
        )
        noise = torch.tensor(librosa.to_mono(y=noise))
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat(
                [
                    noise, 
                    torch.zeros((audio_length-noise_length, ))
                ], 
                dim=-1
            )

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise ) / 2


class MeanNormalizer:
    """Perform mean variance normalization of a tensor."""

    def __call__(self, tensor):
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, unbiased=True, keepdim=True)
        return (tensor - mean) / std


def extract_logmel(
    waveform, 
    sample_rate: int = 16000,
    n_mels: int = 80,
    power = 1.0, # 1 for energy, 2 for power
    to_db_flag: bool = True,
    cmn_flag: bool = True,
    n_fft: int = 400,
    win_length: int = None,
    hop_length: int = 160
):
    # With sample rate 16000 Hz, 1/16000 * 400 = 0.025
    # so n_fft = 400 yields windows of 25 ms 
    # hop_length: frame-shift of 10 ms

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=power, # energy instead of power
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
        window_fn=torch.hamming_window
    )
    cmn = T.SlidingWindowCmn(cmn_window=n_fft)
    to_db = T.AmplitudeToDB(stype="amplitude")

    melspec = mel_spectrogram(waveform)
    if to_db_flag:
        melspec = to_db(melspec)
    if cmn_flag:
        melspec = cmn(melspec)
    return melspec


def copy_audio(row, base_path):
    set_dir = row["Set"] + "/"
    audio_path = row["File"]

    src = base_path + "vox1_dev/" + audio_path
    if not os.path.isfile(src):
        src = base_path + "vox1_test/" + audio_path
    
    dst = base_path + "subset/raw/" + set_dir + audio_path
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


def create_features_from_row(
    row, base_path, rsc, 
    rbn, reverb, babble, random_clip,
    clip_secs, n_mels,
    power = 1.0, # 1 for energy, 2 for power
    to_db_flag = True,
    cmn_flag = True,
    n_fft: int = 400,
    win_length: int = None,
    hop_length: int = 160,
    data_aug: bool = True,
    full_test: bool=True,
    features_dir: str = "features",
    wave_test: bool = True
):
    audio_path = row["File"]
    wav_path = base_path + "vox1_dev/" + audio_path
    if not os.path.isfile(wav_path):
        wav_path = base_path + "vox1_test/" + audio_path
    filename = os.path.splitext(os.path.basename(wav_path))[0]
    waveform, sample_rate = torchaudio.load(wav_path)

    ls = []
    for augment in [
        "none", "speed", "noise", 
        "reverb", "babble"
    ]:
        filename_aug = ""

        if row["Set"] != "train" and augment != "none":
            continue

        if data_aug == False and augment != "none":
            continue

        if augment == "speed":
            waveform = rsc(waveform)   
            filename_aug = "spd"
        elif augment == "noise":
            waveform = rbn(waveform)
            filename_aug = "ns"
        elif augment == "reverb":
            waveform = torch.tensor(
                reverb(
                    waveform,
                    sample_rate=16000
                )
            )
            filename_aug = "rvrb"
        elif augment == "babble":
            waveform = babble(waveform)
            filename_aug = "bbl"

        if clip_secs is not None:
            if not (row["Set"] == "test" and full_test):
                waveform = random_clip(waveform)
        seconds = librosa.get_duration(
            y=waveform[0], sr=sample_rate
        )

        save_path = base_path + f"subset/{features_dir}_{clip_secs}/" \
            + row["Set"] + "/" + row["File"]
        save_dir = os.path.dirname(save_path)

        if features_dir == "verification" or \
            (row["Set"] == "test" and wave_test):
            melspec = waveform
        else:
            melspec = extract_logmel(
                waveform=waveform,  
                sample_rate=sample_rate, 
                n_mels=n_mels,
                power=power,
                to_db_flag=to_db_flag,
                cmn_flag=cmn_flag,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length
            )

        melspec_filename = save_dir + "/" + filename \
            + "_" + filename_aug + ".pt"
        melspec_dir = os.path.dirname(melspec_filename)
        os.makedirs(melspec_dir, exist_ok=True)
        torch.save(melspec, melspec_filename)
    
        relative_path = f"{features_dir}_{clip_secs}/" \
            + row["Set"] + "/" + row["File"]
        relative_dir = os.path.dirname(relative_path)
        relative_filename = relative_dir + "/" + filename \
            + "_" + filename_aug + ".pt"

        ls.append(
            (
                row["Set"], 
                row["Speaker"], 
                "logmel", 
                augment,
                seconds,
                os.path.dirname(row["File"]),
                relative_filename
            )
        )

    return ls


def create_dataset(
    num_speakers: int = 20,
    base_path: str = "E:/Datasets/VoxCeleb1/",
    noise_dir: str = "E:/Datasets/Musan/noise",
    babble_dir: str = "E:/Datasets/Musan/speech",
    clip_secs: int = 3,
    n_mels: int = 80,
    power: float = 1.0, # 1 for energy, 2 for power
    to_db_flag: bool = True,
    cmn_flag: bool = True,
    speaker_ids: list = None,
    n_fft: int = 400,
    win_length: int = None,
    hop_length: int = 160,
    data_aug: bool = True,
    full_test: bool = True,
    wave_test: bool = True
):
    if clip_secs is not None:
        random_clip = RandomClip(clip_secs=clip_secs)
    else:
        random_clip = None
    rsc = RandomSpeedChange()
    rbn = RandomBackgroundNoise(
        noise_dir=noise_dir
    )
    reverb = Pedalboard(
        [Reverb(room_size=0.75)]
    )
    babble = RandomBackgroundNoise(
        noise_dir=babble_dir,
        min_snr_db=15, 
        max_snr_db=20
    )

    os.makedirs(base_path + "subset", exist_ok=True)
    
    # Get the official train, val, test splits
    ls = []
    with open(base_path + "iden_split.txt") as file:
        
        gender_df = pd.read_csv(base_path + "vox1_meta.csv", sep="\t")
        m_ratio = gender_df["Gender"].value_counts(normalize=True)["m"]
        f_ratio = gender_df["Gender"].value_counts(normalize=True)["f"]
        n_males = int(num_speakers * m_ratio)
        n_females = num_speakers - n_males
        
        if speaker_ids is None:
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
        else:
            chosen_ids = speaker_ids

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

    ls = []
    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Creating dataset",
        leave=False
    ):
        # copy_audio(row, base_path)
        feat_ls = create_features_from_row(
            row=row, 
            base_path=base_path,
            rsc=rsc,
            rbn=rbn,
            reverb=reverb,
            babble=babble,
            random_clip=random_clip,
            clip_secs=clip_secs,
            n_mels=n_mels,
            power=power,
            to_db_flag=to_db_flag,
            cmn_flag=cmn_flag,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            data_aug=data_aug,
            full_test=full_test,
            wave_test=wave_test
        )
        ls.extend(feat_ls)

    df = pd.DataFrame(
        ls, 
        columns = [
            "Set", "Speaker", "Type", "Augment", 
            "Seconds", "Path", "File"
        ]
    )

    csv_base_path = base_path + "subset/"

    df.to_csv(
        csv_base_path + f"subset_features_{clip_secs}.csv", 
        index_label=False
    )

    speaker_ids = df["Speaker"].unique()
    label_dict = {speaker_ids[idx]: idx for idx in range(len(speaker_ids))}
    label_df = pd.DataFrame.from_dict(
        label_dict, orient="index", columns=["label"]
    )
    label_df.to_csv(
        csv_base_path + f"subset_labels_{clip_secs}.csv", 
        index_label=False
    )


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
    ylabel="Frequency (Hz)", 
    aspect="auto", 
    xmax=None
):
    """Plot spectrogram function from torchaudio tutorial."""
    sns.set_theme()
    sns.set(style="ticks", font="Times New Roman")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc("axes", titlesize=12)
    plt.rc("figure", titlesize=16)

    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Mel Spectrogram")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("Time (frames)")
    im = axs.imshow(
        spectrogram, 
        origin="lower", 
        aspect=aspect,
        cmap=plt.get_cmap("viridis")
    )
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs, format="%+2.0f dB")
    plt.show(block=False)


def plot_specaug(
    melspec1,
    melspec2,
    ylabel="Frequency",
    aspect="auto"
):
    sns.set_theme()
    sns.set(style="ticks", font="Times New Roman")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc("axes", titlesize=12)
    plt.rc("figure", titlesize=16)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        sharex=True,
        figsize=(6,5)
    )
    fig.suptitle("Log Mel Spectrograms")
    ax1.set_title("Time Masking")
    ax1.set_ylabel(ylabel)
    im = ax1.imshow(
        melspec1, 
        origin="lower", 
        aspect=aspect,
        cmap=plt.get_cmap("viridis")
    )
    ax2.set_title("Frequency Masking")
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel("Time (frames)")
    im = ax2.imshow(
        melspec2, 
        origin="lower", 
        aspect=aspect,
        cmap=plt.get_cmap("viridis")
    )
    fig.colorbar(im, ax=[ax1, ax2], format="%+2.0f dB")
    plt.show(block=False)
    fig.savefig("spec_aug.png", dpi=300)


def pad_tensor(x, x_len, max_len):
    if x_len < max_len:
        x = x[0]  
        pad_begin_len = torch.randint(
            low=0, high=max_len - x_len, size=(1,)
        )
        pad_end_len = max_len - x_len - pad_begin_len

        pad_begin = torch.zeros((x.shape[0], pad_begin_len))
        pad_end = torch.zeros((x.shape[0], pad_end_len))
        try:
            padded_tensor = torch.cat((pad_begin, x, pad_end), 1)
        except Exception as e:
            print("pad_begin.shape", pad_begin.shape)
            print("x.shape", x.shape)
            print("pad_end.shape", pad_end.shape)
            raise e

        return padded_tensor.unsqueeze(0)
    else:
        return x


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
        wav_ls[-1] = pad_tensor(
            wav_ls[-1], wav_ls[-1].shape[1], num_samples
        )

    return wav_ls


def get_data(
    set_name, 
    base_path, 
    num_secs, 
    random_clip,
    label_dict,
    wave_data,
    limited_train=None
):
    df = pd.read_csv(
        base_path + f"subset/subset_features_{num_secs}.csv"
    )
    df = df.sample(frac=1).reset_index(drop=True)
    df_set = df[df["Set"] == set_name]
    speaker_dict = dict()

    features_ls = []
    y_ls = []
    for idx, row in df_set.iterrows():
        speaker = row["Speaker"]
        speaker_dict.setdefault(speaker, 0)
        
        if row["Augment"] != "none":
            continue

        if set_name == "train" and limited_train is not None \
            and speaker_dict[speaker] >= limited_train:
            continue
        speaker_dict[speaker] += 1

        features = torch.load(base_path + f"subset/{row['File']}")
        if wave_data:
            features = random_clip(features)
            features = extract_logmel(features)
        features_ls.append(
            features.numpy()
        )
        y_ls.append(
            label_dict[row["Speaker"]]
        )
    X = np.vstack(features_ls)
    X = X.reshape(X.shape[0], -1)
    y = np.vstack(y_ls).squeeze(-1)

    return X, y


def kmeans_plot(
    base_path: str = "E:/Datasets/VoxCeleb1/",
    num_secs: int = 4,
    num_speakers: int = 100,
    limited_train: int = 10
):
    random_clip = RandomClip(clip_secs=num_secs)
    label_dict = pd.read_csv(
        base_path + f"subset/subset_labels_{num_secs}.csv"
    ).to_dict()["label"]

    sns.set_theme()
    sns.set(style="ticks", font="Times New Roman")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc("axes", titlesize=12)
    plt.rc("figure", titlesize=16)
    # plt.rcParams["font.size"] = 16

    X_train, y_train = get_data(
        set_name="train",
        base_path=base_path,
        num_secs=num_secs,
        random_clip=random_clip,
        label_dict=label_dict,
        limited_train=limited_train,
        wave_data=False
    )
    print("Training shape:", X_train.shape)
    X_test, y_test = get_data(
        set_name="test",
        base_path=base_path,
        num_secs=num_secs,
        random_clip=random_clip,
        label_dict=label_dict,
        wave_data=True
    )
    print("Test shape:", X_test.shape)

    kmeans = KMeans(n_clusters=num_speakers)
    kmeans.fit(X_train)

    pca = PCA(2)
    pca.fit(X_train)
    Xpca = pca.transform(X_test)

    preds = kmeans.predict(X_test)
    u_preds = np.unique(preds)
    u_y = np.unique(y_test)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(10,4),
        sharey=True
    )
    fig.suptitle("Test set label distribution")
    ax1.set_title("K-Means clusters")
    ax2.set_title("Ground truth")

    for label in u_preds:
        ax1.scatter(
            Xpca[preds == label, 0], 
            Xpca[preds == label, 1], 
            # label=label,
            alpha=1,
            linewidths=0.5,
            edgecolors="black"
        )

    for label in u_y:
        ax2.scatter(
            Xpca[y_test == label, 0], 
            Xpca[y_test == label, 1], 
            # label=label,
            alpha=1,
            linewidths=0.5,
            edgecolors="black"
        )
    plt.show()
    fig.savefig("k_means.png", dpi=300)

    return X_test, Xpca, y_test, preds


def get_stats_lists(df, meta_df, csv_base_path):
    nat_ls = []
    gender_ls = []
    seconds_ls = []
    for idx, row in df.iterrows():
        augment = row["Augment"]
        if augment == "none":
            speaker = row["Speaker"]
            nationality = list(
                meta_df[meta_df["VoxCeleb1 ID"] == speaker]["Nationality"]
            )[0]
            gender = list(
                meta_df[meta_df["VoxCeleb1 ID"] == speaker]["Gender"]
            )[0]
            nat_ls.append(nationality)
            gender_ls.append(gender)
            
            path = row["Path"]
            basename = os.path.basename(
                row["File"]
            )[0:-4]
            file_path = csv_base_path + "vox1_dev/" + \
                path + f"/{basename}.wav"
            if not os.path.exists(file_path):
                file_path = csv_base_path + "vox1_test/" + \
                    path + f"/{basename}.wav"
            waveform, sample_rate = torchaudio.load(file_path)
            seconds = librosa.get_duration(
                y=waveform[0], sr=sample_rate
            )
            seconds_ls.append(seconds)

    return pd.Series(nat_ls), pd.Series(gender_ls), pd.Series(seconds_ls)


def print_stats(subset, df, meta_df, csv_base_path):
    print("*" * 20)
    print(f"{subset.upper()} SET")
    print("*" * 20)
    print(f"Number of samples: {len(df)}")
    nat_series, gender_series, seconds_series = get_stats_lists(
        df, meta_df, csv_base_path
    )
    print("Nationality:")
    print(nat_series.value_counts(normalize=True).round(2))
    print("Gender:")
    print(gender_series.value_counts(normalize=True).round(2))
    print("Seconds:")
    print(f"Mean: {seconds_series.mean().round(2)}; Std: {seconds_series.std().round(2)}")


def get_dataset_stats(
    base_path: str = "E:/Datasets/VoxCeleb1/",
    num_secs: int = 4,
    num_speakers: int = 100
):
    print("Starting")
    df = pd.read_csv(
        base_path + f"subset/subset_features_{num_secs}.csv"
    )
    df_no_aug = df[df["Augment"] == "none"]
    meta_df = pd.read_csv(base_path + "vox1_meta.csv", sep="\t")
    label_dict = pd.read_csv(
        base_path + f"subset/subset_labels_{num_secs}.csv"
    ).to_dict()["label"]
    print("Got the dataframes")

    speaker_ids = df["Speaker"].unique()
    assert len(speaker_ids) == num_speakers
    
    tot_seconds_ls = []

    with open(base_path + "iden_split.txt") as file:
        all_lines = file.readlines()
        num_lines = len(all_lines)
        print("To analyze entire dataset")
        for line in tqdm(
            all_lines,
            total=num_lines,
            desc="Analyzing entire dataset"
        ):
            set_num, audio_path = line.split()
            full_path = base_path + "vox1_dev/" + audio_path
            if not os.path.isfile(full_path):
                full_path = base_path + "vox1_test/" + audio_path
            waveform, sample_rate = torchaudio.load(
                full_path
            )
            seconds = librosa.get_duration(
                y=waveform[0],
                sr=sample_rate
            )
            tot_seconds_ls.append(seconds)

    print("Finished analysis of entire dataset")

    tot_seconds_ls = np.array(tot_seconds_ls)
    tot_speaker_ids = meta_df["VoxCeleb1 ID"].unique()

    gender_ls = []
    nat_ls = []
    for speaker_id in speaker_ids:
        gender_ls.append(
            list(
                meta_df[meta_df["VoxCeleb1 ID"] == speaker_id]["Gender"]
            )[0]
        )
        nat_ls.append(
            list(
                meta_df[meta_df["VoxCeleb1 ID"] == speaker_id]["Nationality"]
            )[0]
        )

    verif_df = pd.read_csv(
        base_path + f"subset/subset_verification.csv"
    )
    verif_speaker_ids = verif_df["Speaker"].unique()
    verif_seconds = verif_df["Seconds"]
    verif_gender_ls = []
    verif_nat_ls = []
    for speaker_id in verif_speaker_ids:
        verif_gender_ls.append(
            list(
                meta_df[meta_df["VoxCeleb1 ID"] == speaker_id]["Gender"]
            )[0]
        )
        verif_nat_ls.append(
            list(
                meta_df[meta_df["VoxCeleb1 ID"] == speaker_id]["Nationality"]
            )[0]
        )

    print("*" * 20)
    print("GENERAL STATS")
    print("*" * 20)
    print("Samples in entire dataset:", num_lines)
    print("Samples in identification subset (without augment):", len(df_no_aug))
    print("Samples in verification subset:", len(verif_df))
    print("Speakers in entire dataset:", len(tot_speaker_ids))
    print("Speakers in identification subset:", len(speaker_ids))
    print("Speakers in verification subset:", len(verif_speaker_ids))
    print("Gender in entire dataset:")
    print(meta_df["Gender"].value_counts(normalize=True).round(2))
    print("Gender in identification subset:")
    print(pd.Series(gender_ls).value_counts(normalize=True).round(2))
    print("Gender in verification subset:")
    print(pd.Series(verif_gender_ls).value_counts(normalize=True).round(2))
    print("Nationality in entire dataset:")
    print(meta_df["Nationality"].value_counts(normalize=True).round(2))
    print("Nationality in identification subset:")
    print(pd.Series(nat_ls).value_counts(normalize=True).round(2))
    print("Nationality in verification subset:")
    print(pd.Series(verif_nat_ls).value_counts(normalize=True).round(2))
    print("Mean seconds in entire dataset:", round(np.mean(tot_seconds_ls), 2))
    print("Std seconds in entire dataset:", round(np.std(tot_seconds_ls), 2))
    print("Mean seconds in verification dataset:", verif_seconds.mean().round(2))
    print("Std seconds in verification dataset:", verif_seconds.std().round(2))

    df_train = df[df["Set"] == "train"]
    df_train = df_train[df_train["Augment"] == "none"]
    df_test = df[df["Set"] == "test"]
    df_test = df_test[df_test["Augment"] == "none"]

    print(
        "Subset now counting samples instead" 
        "of speakers (this is the reason why some "
        "numbers are different)"
    )
    print_stats("subset", df, meta_df, base_path)
    print_stats("train", df_train, meta_df, base_path)
    print_stats("test", df_test, meta_df, base_path)


def create_mfcc(
    base_path: str = "E:/Datasets/VoxCeleb1/",
    num_secs: int = 4,
    data_aug: bool = True
):
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=30
    )

    random_clip = RandomClip(clip_secs=num_secs)

    df = pd.read_csv(
        base_path + f"subset/subset_features_{num_secs}.csv"
    )
    df = df[df["Augment"] == "none"]
    label_dict = pd.read_csv(
        base_path + f"subset/subset_labels_{num_secs}.csv"
    ).to_dict()["label"]

    res_ls = []

    for idx, row in df.iterrows():
        path = row["Path"]
        basename = os.path.basename(
            row["File"]
        )[0:-4]
        file_path = base_path + "vox1_dev/" + \
            path + f"/{basename}.wav"
        if not os.path.exists(file_path):
            file_path = base_path + "vox1_test/" + \
                path + f"/{basename}.wav"
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = random_clip(waveform)
        mfcc = mfcc_t(waveform)
        save_dir = base_path + f"subset/mfcc_{num_secs}/" + \
            path + f"/{basename}.pt"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(mfcc, save_dir)

        file_dir = f"subset/mfcc_{num_secs}/" + \
            path + f"/{basename}.pt"

        res_ls.append(
            (
                row["Set"],
                row["Speaker"],
                "mfcc",
                "none",
                path,
                file_dir,
                label_dict[row["Speaker"]]
            )
        )

    df_res = pd.DataFrame(
        res_ls, 
        columns =["Set", "Speaker","Type", "Augment", "Path", "File", "Label"]
    )
    df_res.to_csv(
        base_path + f"subset/subset_mfcc_{num_secs}.csv",
        index_label=False
    )


def train_sklearn_model(
    model,
    num_secs = 4,
    base_path: str = "E:/Datasets/VoxCeleb1/",
    validate: bool = False,
    limited_train: int = 10
):
    random_clip = RandomClip(clip_secs=num_secs)
    label_dict = pd.read_csv(
        base_path + f"subset/subset_labels_{num_secs}.csv"
    ).to_dict()["label"]

    X_train, y_train = get_data(
        set_name="train",
        base_path=base_path,
        num_secs=num_secs,
        random_clip=random_clip,
        label_dict=label_dict,
        limited_train=limited_train,
        wave_data=False
    )
    print("Loaded train", X_train.shape)
    if validate:
        X_val, y_val = get_data(
            set_name="val",
            base_path=base_path,
            num_secs=num_secs,
            random_clip=random_clip,
            label_dict=label_dict,
            wave_data=False
        )
        print("Loaded val", X_val.shape)
    X_test, y_test = get_data(
        set_name="test",
        base_path=base_path,
        num_secs=num_secs,
        random_clip=random_clip,
        label_dict=label_dict,
        wave_data=True
    )
    print("Loaded test", X_test.shape)

    model.fit(X_train, y_train)
    print("Finished training")
    if validate:
        return model.predict(X_val), y_val
    else:
        return model.predict(X_test), y_test


def create_verification_dataset(
    speaker_ids,
    chosen_ids = None,
    num_speakers: int = 10,
    base_path: str = "E:/Datasets/VoxCeleb1/",
    n_mels: int = 80,
    power: float = 1.0, # 1 for energy, 2 for power
    to_db_flag: bool = True,
    cmn_flag: bool = True,
    n_fft: int = 400,
    win_length: int = None,
    hop_length: int = 160,
    data_aug: bool = False,
    full_test: bool=True
):
    ls = []
    with open(base_path + "iden_split.txt") as file:
        
        gender_df = pd.read_csv(base_path + "vox1_meta.csv", sep="\t")
        m_ratio = gender_df["Gender"].value_counts(normalize=True)["m"]
        f_ratio = gender_df["Gender"].value_counts(normalize=True)["f"]
        n_males = int(num_speakers * m_ratio)
        n_females = num_speakers - n_males

        if chosen_ids is None:
            restricted_df = gender_df[~gender_df["VoxCeleb1 ID"].isin(speaker_ids)]
            
            male_ids = random.sample(
                list(
                    restricted_df[restricted_df["Gender"] == "m"]["VoxCeleb1 ID"].unique()
                ),
                n_males
            )
            female_ids = random.sample(
                list(
                    restricted_df[restricted_df["Gender"] == "f"]["VoxCeleb1 ID"].unique()
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

    ls = []
    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Creating verification dataset",
        leave=False
    ):
        # copy_audio(row, base_path)
        feat_ls = create_features_from_row(
            row=row, 
            base_path=base_path,
            rsc=None,
            rbn=None,
            reverb=None,
            babble=None,
            random_clip=None,
            clip_secs=None,
            n_mels=n_mels,
            power=power,
            to_db_flag=to_db_flag,
            cmn_flag=cmn_flag,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            data_aug=data_aug,
            features_dir="verification"
        )
        ls.extend(feat_ls)

    df = pd.DataFrame(
        ls, 
        columns = [
            "Set", "Speaker", "Type", "Augment", 
            "Seconds", "Path", "File"
        ]
    )

    csv_base_path = base_path + "subset/"

    df.to_csv(
        csv_base_path + f"subset_verification.csv", 
        index_label=False
    )
