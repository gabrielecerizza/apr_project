import pandas as pd
import torch
import torchaudio.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.utils import pad_tensor


class VoxCelebDataset(Dataset):
    def __init__(
        self, 
        csv_base_path: str = "E:/Datasets/VoxCeleb1/subset/", 
        set_name: str = "train",
        feat_type: str = "logmel",
        num_secs: int = 3,
        spec_augment: bool = True,
        from_memory: bool = False
    ):
        super(VoxCelebDataset, self).__init__()

        self.set_name = set_name.lower()
        self.type = feat_type.lower()
        self.num_secs = num_secs
        self.spec_augment = spec_augment
        self.from_memory = from_memory

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
                self.data.append(
                    (torch.load(row["File"]), row["Speaker"])
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
            filename = self.df.iloc[idx]["File"]
            features = torch.load(filename)
            speaker_id = self.df.iloc[idx]["Speaker"]

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
        x_len = x.shape[2]
        if x_len < max_len:
            print("padding inside dataloader")
            print("x shape", x.shape)
            print("x_len", x_len)
            print("max_len", max_len)
            padded_tensor = pad_tensor(x, x_len, max_len)
            new_features_batch.append(padded_tensor)
        else:
            new_features_batch.append(x)

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
        pin_memory: bool = True,
        num_workers: int = 0,
        spec_augment: bool = True,
        from_memory: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.spec_augment = spec_augment
        self.from_memory = from_memory

    def prepare_data(self):
        # Download
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.vox_train = VoxCelebDataset(
                set_name="train",
                spec_augment=self.spec_augment,
                from_memory=self.from_memory
            )
            self.vox_val = VoxCelebDataset(
                set_name="val",
                spec_augment=self.spec_augment,
                from_memory=self.from_memory
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.vox_test = VoxCelebDataset(
                set_name="test",
                spec_augment=self.spec_augment,
                from_memory=self.from_memory
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