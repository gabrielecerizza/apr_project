import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import pad_tensor


class VoxCelebDataset(Dataset):
    def __init__(
        self, 
        csv_base_path: str = "E:/Datasets/VoxCeleb1/subset/", 
        set_name: str = "train",
        feat_type: str = "logmel"
    ):
        self.set_name = set_name.lower()
        self.type = feat_type.lower()
        self.df = pd.read_csv(csv_base_path + "subset_features.csv")
        self.df = self.df[self.df["Set"] == self.set_name]
        self.df = self.df[self.df["Type"] == self.type]
        
        self.label_dict = pd.read_csv(
            csv_base_path + "subset_labels.csv"
        ).to_dict()["label"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.df.iloc[idx]["File"]
        speaker_id = self.df.iloc[idx]["Speaker"]
        sample = {
            "features": torch.load(filename),
            "label": torch.tensor(self.label_dict[speaker_id])
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
        x_len = x.shape[1]
        if x_len < max_len:
            padded_tensor = pad_tensor(x, x_len, max_len)
            new_features_batch.append(padded_tensor)
        else:
            new_features_batch.append(x)

    features = torch.stack(new_features_batch)
    labels = torch.stack([x["label"] for x in batch])

    return {
        "features": features,
        "labels": labels
    }


class VoxCelebDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "E:/Datasets/VoxCeleb1/",
        batch_size: int = 4,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory

    def prepare_data(self):
        # Download
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.vox_train = VoxCelebDataset(set_name="train")
            self.vox_val = VoxCelebDataset(set_name="val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.vox_test = VoxCelebDataset(set_name="test")

    def train_dataloader(self):
        return DataLoader(
            self.vox_train, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate_vox
        )

    def val_dataloader(self):
        return DataLoader(
            self.vox_val, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=collate_vox
        )

    def test_dataloader(self):
        return DataLoader(
            self.vox_test, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=collate_vox
        )