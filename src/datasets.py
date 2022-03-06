import random

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from pedalboard import Pedalboard, Reverb
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .utils import (
    pad_tensor, RandomBackgroundNoise, RandomSpeedChange,
    extract_logmel
)


class VoxCelebDataset(Dataset):
    def __init__(
        self, 
        csv_base_path: str = "E:/Datasets/VoxCeleb1/subset/",
        noise_dir: str = "E:/Datasets/Musan/noise",
        babble_dir: str = "E:/Datasets/Musan/speech", 
        set_name: str = "train",
        feat_type: str = "logmel",
        num_secs: int = 3,
        spec_augment: bool = True,
        data_augment: bool = False,
        from_memory: bool = False,
        sample_rate: int = 16000,
        n_mels: int = 80,
        power: float = 1.0, # 1 for energy, 2 for power
        to_db_flag: bool = True,
        cmn_flag: bool = True,
        n_fft: int = 400,
        win_length: int = None,
        hop_length: int = 160
    ):
        super(VoxCelebDataset, self).__init__()

        self.set_name = set_name.lower()
        self.type = feat_type.lower()
        self.num_secs = num_secs
        self.spec_augment = spec_augment
        self.data_augment = data_augment
        self.from_memory = from_memory
        self.csv_base_path = csv_base_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.power = power
        self.to_db_flag = to_db_flag
        self.cmn_flag = cmn_flag
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        if self.data_augment:
            self.rsc = RandomSpeedChange()
            self.rbn = RandomBackgroundNoise(
                noise_dir=noise_dir
            )
            self.reverb = Pedalboard(
                [Reverb(room_size=0.75)], 
                # sample_rate=16000
            )
            self.babble = RandomBackgroundNoise(
                noise_dir=babble_dir,
                min_snr_db=15, 
                max_snr_db=20
            )

        self.df = pd.read_csv(
            csv_base_path + f"subset_features_{num_secs}.csv"
        )
        self.df = self.df[self.df["Set"] == self.set_name]
        self.df = self.df[self.df["Type"] == self.type]
        
        self.label_dict = pd.read_csv(
            csv_base_path + f"subset_labels_{num_secs}.csv"
        ).to_dict()["label"]

        if self.from_memory:
            self.data = []

            for idx, row in tqdm(
                self.df.iterrows(),
                total=len(self.df),
                desc=f"Loading data for {set_name}"
            ):
                filename = self.csv_base_path + row["File"]
                self.data.append(
                    (torch.load(filename), row["Speaker"])
                )
        else:
            self.data = None

        self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
        self.time_masking = T.TimeMasking(time_mask_param=5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.from_memory:
            features = self.data[idx][0]
            speaker_id = self.data[idx][1]
        else:
            filename = self.csv_base_path + self.df.iloc[idx]["File"]
            features = torch.load(filename)
            speaker_id = self.df.iloc[idx]["Speaker"]

        if self.data_augment and self.set_name == "train":
            augtype = random.randint(0,4)
            if augtype == 1:
                features = self.rsc(features)
            elif augtype == 2:
                features = self.rbn(features)
            elif augtype == 3:
                features = torch.tensor(
                    self.reverb(
                        features,
                        sample_rate=self.sample_rate
                    )
                )
            elif augtype == 4:
                features = self.babble(features)

        if self.data_augment:
            features = extract_logmel(
                waveform=features,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                power=self.power,
                to_db_flag=self.to_db_flag,
                cmn_flag=self.cmn_flag,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length
            )

        if self.spec_augment and self.set_name == "train":
            features = self.freq_masking(features)
            features = self.time_masking(features)

        sample = {
            "features": features,
            "labels": torch.tensor(self.label_dict[speaker_id]),
            "speakers": speaker_id
        }

        return sample


def collate_vox(batch):
    """Function to handle feature tensors of different sizes in
    a batch. The tensors are padded with zero (silence). We
    add padding both at the beginning and at the end of the
    tensor, choosing random endpoints.

    Zero padding to fix tensors of varying lengths is also 
    done in Tacotron2 (NVIDIA):
        https://github.com/NVIDIA/tacotron2/blob/master/data_utils.py
    """
    features_batch = [x["features"] for x in batch]
    max_len = torch.max(
        torch.tensor([x.shape[-1] for x in features_batch])
    )

    new_features_batch = []

    for x in features_batch:
        x_len = x.shape[-1]
        new_features_batch.append(pad_tensor(x, x_len, max_len))

    features = torch.stack(new_features_batch)
    labels = torch.stack([x["labels"] for x in batch])

    return {
        "features": features,
        "labels": labels,
        "speakers": [x["speakers"] for x in batch]
    }


class VoxCelebDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "E:/Datasets/VoxCeleb1/",
        batch_size: int = 4,
        num_secs: int =3,
        pin_memory: bool = True,
        num_workers: int = 0,
        spec_augment: bool = True,
        from_memory: bool = False,
        data_augment: bool = False,
        sample_rate: int = 16000,
        n_mels: int = 80,
        power: float = 1.0, # 1 for energy, 2 for power
        to_db_flag: bool = True,
        cmn_flag: bool = True,
        n_fft: int = 400,
        win_length: int = None,
        hop_length: int = 160
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_secs = num_secs
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.spec_augment = spec_augment
        self.data_augment = data_augment
        self.from_memory = from_memory
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.power = power
        self.to_db_flag = to_db_flag
        self.cmn_flag = cmn_flag
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def prepare_data(self):
        # Download
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.vox_train = VoxCelebDataset(
                csv_base_path=self.data_dir + "subset/",
                set_name="train",
                spec_augment=self.spec_augment,
                data_augment=self.data_augment,
                from_memory=self.from_memory,
                num_secs=self.num_secs,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                power=self.power,
                to_db_flag=self.to_db_flag,
                cmn_flag=self.cmn_flag,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length
            )
            self.vox_val = VoxCelebDataset(
                csv_base_path=self.data_dir + "subset/",
                set_name="val",
                spec_augment=self.spec_augment,
                data_augment=self.data_augment,
                from_memory=self.from_memory,
                num_secs=self.num_secs,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                power=self.power,
                to_db_flag=self.to_db_flag,
                cmn_flag=self.cmn_flag,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.vox_test = VoxCelebDataset(
                csv_base_path=self.data_dir + "subset/",
                set_name="test",
                spec_augment=self.spec_augment,
                data_augment=self.data_augment,
                from_memory=self.from_memory,
                num_secs=self.num_secs,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                power=self.power,
                to_db_flag=self.to_db_flag,
                cmn_flag=self.cmn_flag,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.vox_train, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_vox
        )

    def val_dataloader(self):
        return DataLoader(
            self.vox_val, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_vox
        )

    def test_dataloader(self):
        return DataLoader(
            self.vox_test, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_vox
        )