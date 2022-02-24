import math
import os
import pathlib
import random

import librosa
import numpy as np
import pandas as pd
import torch, torchaudio
import torchaudio.transforms as T
from pedalboard import Pedalboard, Reverb

from src.utils import pad_tensor


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
            audio_data, speed_factor
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
        noise = librosa.resample(noise[0].numpy(), noise_sr, self.sample_rate)
        noise = torch.tensor(librosa.to_mono(noise))
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


def extract_logmel(waveform, sample_rate, n_mels):
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
        power=1.0, # energy instead of power
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    cmn = T.SlidingWindowCmn(cmn_window=n_fft)
    to_db = T.AmplitudeToDB(stype="amplitude")

    melspec = cmn(to_db(mel_spectrogram(waveform)))
    return melspec


def create_features(
    csv_base_path: str = "E:/Datasets/VoxCeleb1/subset/",
    n_mels: int = 80,
    clip_secs: int = 3
):
    random_clip = RandomClip(clip_secs=clip_secs)
    rsc = RandomSpeedChange()
    rbn = RandomBackgroundNoise()
    reverb = Pedalboard(
        [Reverb(room_size=0.75)], 
        sample_rate=16000
    )
    babble = RandomBackgroundNoise(
        noise_dir="E:/Datasets/Musan/speech",
        min_snr_db=15, 
        max_snr_db=20
    )
    
    df = pd.read_csv(csv_base_path + "subset.csv")

    ls = []
    for index, row in df.iterrows():
        base_path = "E:/Datasets/VoxCeleb1/subset/raw/" \
             + row["Set"] + "/"
        wav_path = base_path + row["File"] 
        filename = os.path.splitext(os.path.basename(wav_path))[0]
        waveform, sample_rate = torchaudio.load(wav_path)

        for augment in [
            "none", "speed", "noise", 
            "reverb", "babble"
        ]:
            filename_aug = ""

            if row["Set"] != "train" and augment != "none":
                continue

            if augment == "speed":
                try:
                    waveform = rsc(waveform)
                except Exception as e:
                    print("Error in speed")
                    print(index)
                    print(wav_path)
                    print(waveform.shape)
                    print(waveform)
                    print(e)
                    
                filename_aug = "spd"
            elif augment == "noise":
                waveform = rbn(waveform)
                filename_aug = "ns"
            elif augment == "reverb":
                waveform = torch.tensor(reverb(waveform))
                filename_aug = "rvrb"
            elif augment == "babble":
                waveform = babble(waveform)
                filename_aug = "bbl"

            waveform = random_clip(waveform)
            seconds = librosa.get_duration(waveform[0], sr=sample_rate)

            save_path = f"E:/Datasets/VoxCeleb1/subset/features_{clip_secs}/" \
                + row["Set"] + "/" + row["File"]
            save_dir = os.path.dirname(save_path)

            melspec = extract_logmel(
                waveform=waveform, 
                sample_rate=sample_rate, 
                n_mels=n_mels
            )

            melspec_filename = save_dir + "/" + filename \
                + "_" + filename_aug + ".pt"
            melspec_dir = os.path.dirname(melspec_filename)
            os.makedirs(melspec_dir, exist_ok=True)
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
            torch.save(melspec, melspec_filename)

    df = pd.DataFrame(
        ls, 
        columns = [
            "Set", "Speaker", "Type", "Augment", 
            "Seconds", "Path", "File"
        ]
    )
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