import math
import os
import pathlib
import random
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from pedalboard import Pedalboard, Reverb
from tqdm.auto import tqdm


class RandomClip:
    def __init__(
        self, 
        sample_rate: int = 16000,
        clip_secs: int = 3
    ):
        self.clip_length = clip_secs * sample_rate

    def __call__(self, audio_data):
        audio_data = audio_data[0]
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[offset:(offset+self.clip_length)]
        elif audio_length < self.clip_length:
            audio_data = pad_tensor(
                audio_data.unsqueeze(0), audio_length, self.clip_length
            )[0]

        return audio_data.unsqueeze(0)


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
    sample_rate,
    n_mels,
    power = 1.0, # 1 for energy, 2 for power
    to_db_flag = True,
    cmn_flag = True
):
    # With sample rate 16000 Hz, 1/16000 * 400 = 0.025
    # so n_fft = 400 yields windows of 25 ms
    n_fft = 400 
    win_length = None
    hop_length = 160 # frame-shift of 10 ms
    n_mels = n_mels

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
    cmn_flag = True
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

        waveform = random_clip(waveform)
        seconds = librosa.get_duration(
            y=waveform[0], sr=sample_rate
        )

        save_path = base_path + f"subset/features_{clip_secs}/" \
            + row["Set"] + "/" + row["File"]
        save_dir = os.path.dirname(save_path)

        melspec = extract_logmel(
            waveform=waveform, 
            sample_rate=sample_rate, 
            n_mels=n_mels,
            power=power,
            to_db_flag=to_db_flag,
            cmn_flag=cmn_flag
        )

        melspec_filename = save_dir + "/" + filename \
            + "_" + filename_aug + ".pt"
        melspec_dir = os.path.dirname(melspec_filename)
        os.makedirs(melspec_dir, exist_ok=True)
        torch.save(melspec, melspec_filename)
    
        ls.append(
            (
                row["Set"], 
                row["Speaker"], 
                "logmel", 
                augment,
                seconds,
                os.path.dirname(row["File"]),
                melspec_filename
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
    cmn_flag: bool = True
):
    random_clip = RandomClip(clip_secs=clip_secs)
    rsc = RandomSpeedChange()
    rbn = RandomBackgroundNoise(
        noise_dir=noise_dir
    )
    reverb = Pedalboard(
        [Reverb(room_size=0.75)], 
        # sample_rate=16000
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
    # df.to_csv(base_path + "subset/subset.csv", index_label=False)

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
            cmn_flag=cmn_flag
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



    