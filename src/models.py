import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_distances

from src.losses import AAMSoftmaxLoss
from src.metrics import (
    compute_eer, compute_error_rates, compute_min_dcf
)


class SpeakerRecognitionModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        embeddings_dim: int = 512,
        loss_margin: float = 0.5,
        loss_scale: float = 64,
        average: str = "weighted",
        num_secs: int = 3,
        base_path: str = "E:/Datasets/VoxCeleb1/subset/",
        feature_type: str = "logmel",
        embeddings_strategy: str = "separate",
        top_n: int = 100
    ) -> None:
        super(SpeakerRecognitionModel, self).__init__()

        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.loss_margin = loss_margin
        self.loss_scale = loss_scale
        self.average = average
        self.num_secs = num_secs
        self.base_path = base_path
        self.feature_type = feature_type
        self.embeddings_strategy = embeddings_strategy
        self.top_n = top_n
        self.loss_func = AAMSoftmaxLoss(
            num_classes=num_classes,
            embeddings_dim=embeddings_dim,
            margin=loss_margin,
            scale=loss_scale
        )

        self.acc = Accuracy(
            num_classes=self.num_classes,
            average=self.average
        )

        self.f1 = F1Score(
            num_classes=self.num_classes,
            average=self.average
        )

    def create_embeddings(self):
        embeddings_ls = []

        df = pd.read_csv(
            self.base_path + f"subset_features_{self.num_secs}.csv"
        )
        df = df[df["Type"] == self.feature_type]
        for index, row in df.iterrows():
            # We compute embeddings only for
            # the original files
            if row["Augment"] != "none":
                continue

            file = row["File"]
            filename = os.path.splitext(os.path.basename(file))[0]
            features = torch.load(file).unsqueeze(1)
            embeddings = self(features)[0]

            if self.embeddings_strategy == "mean":
                raise NotImplementedError
            elif self.embeddings_strategy == "separate":
                embeddings_file = self.base_path + "embeddings/" + row["Path"] \
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
            self.base_path + f"subset_embeddings_{self.num_secs}.csv", 
            index_label=False
        )

    def compute_scores(self, batch):
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
            self.base_path + f"subset_embeddings_{self.num_secs}.csv"
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
            e_distances = np.sort(e_distances)[:self.top_n]

            me = np.mean(e_distances)
            se = np.std(e_distances)

            for idx, test_speaker in batch["speakers"]:
                test_embedding = batch["embeddings"][idx]

                distance = cosine_distances([embedding], [test_embedding])[0]

                t_distances = cosine_distances([test_embedding], cohort)[0]
                t_distances = np.sort(t_distances)[:self.top_n]

                mt = np.mean(t_distances)
                st = np.std(t_distances)

                e_term = (distance - me) / se
                t_term = (distance - mt) / st
                score = 0.5 * (e_term + t_term)

                scores.append(score)
                labels.append(int(speaker == test_speaker))

        return scores, labels

    def training_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        x = x.unsqueeze(1)
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)

        train_acc = self.acc(logits, true_labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.create_embeddings()

        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        x = x.unsqueeze(1)
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)

        val_acc = self.acc(logits, true_labels)

        scores, labels = self.compute_scores(batch)
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        val_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        val_eer = compute_eer(scores, labels)

        metrics_ls = [
            ("val_loss", loss),
            ("val_acc", val_acc), 
            ("val_min_dcf", val_min_dcf),
            ("val_eer", val_eer)
        ]
        for metric_name, metric_name in metrics_ls:
            self.log(
                metric_name, 
                metric_name, 
                prog_bar=True, 
                on_step=True, 
                on_epoch=True
            )

        return loss

    def on_test_epoch_start(self) -> None:
        self.create_embeddings()

        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        scores, labels = self.compute_scores(batch)
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        test_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        test_eer = compute_eer(scores, labels)

        metrics_ls = [ 
            ("test_min_dcf", test_min_dcf),
            ("test_eer", test_eer)
        ]
        for metric_name, metric_name in metrics_ls:
            self.log(
                metric_name, 
                metric_name, 
                prog_bar=True, 
                on_step=True, 
                on_epoch=True
            )

        return scores, labels

    def test_epoch_end(self, outputs):
        model_name = str(type(self)).lower()
        save_dir = "results/"
        os.makedirs(save_dir, exist_ok=True)

        scores, labels = [], []

        for tup in outputs:
            scores.extend(tup[0])
            labels.extend(tup[1])

        fnrs, fprs, thresholds = compute_error_rates(
            np.array(scores), np.array(labels)
        )
        min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        eer = compute_eer(scores, labels)

        res = {
            "min_dcf": min_dcf,
            "eer": eer,
            "model_name": model_name,
            "embeddings_dim": self.embeddings_dim,
            "loss": self.loss_func.__class__.__name__,
            "loss_margin": self.loss_margin,
            "loss_scale": self.loss_scale,
            "average": self.average,
            "num_secs": self.num_secs,
            "feature_type": self.feature_type,
            "embeddings_strategy": self.embeddings_strategy,
            "top_n": self.top_n
        }

        with open(
            f"{save_dir}/{model_name}_steps={self.train_steps}.json", 
            "w", encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)

    def setup(self, stage):
        if stage == "fit":
            total_devices = self.hparams.n_gpus * self.hparams.n_nodes
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.hparams.epochs * train_batches) // \
                self.hparams.accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=2e-5, 
            eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }


class TDNNLayer(nn.Module):
    """Time delay neural network layer, as described in [1].

    Since we handle only symmetric contexts, we exploit 1d
    convolution in the implementation, which was suggested
    as a fast and memory-efficient solution in: 
        https://github.com/yuyq96/D-TDNN

    References
    ----------
        [1] V. Peddinti, D. Povey and Sanjeev Khudanpur,
        "A time delay neural network architecture for 
        efficient modeling of long temporal contexts",
        in Proc. Interspeech 2015, 2015.
    """
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        context_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
    ) -> None:
        """Use context_size and dilation to reproduce
        the contexts referred to in the papers.

        e.g.    context_size = 5
                dilation = 1
                context = [-2,-1,0,1,2]

                context_size = 5
                dilation = 2
                context = [-4,-2,0,2,4]

                context_size = 3
                dilation = 2
                context = [-2,0,2]
        """
        super(TDNNLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=context_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class DenseLayer(nn.Module):
    """Simple fully connected layer with ReLU 
    activation.
    """
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 512
    ) -> None:
        super(DenseLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(
                in_features=in_features, 
                out_features=out_features
            ),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class StatsPoolingLayer(nn.Module):
    """Layer that computes the mean and standard
    deviation of the input tensor and returns
    the tensor resulting from their concatenation.
    """
    def __init__(self) -> None:
        super(StatsPoolingLayer, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)


class TDNN(nn.Module):
    """Time delay neural network for x-vectors computation,
    as described in [1] and [2].

    References
    ----------
        [1] V. Peddinti, D. Povey and S. Khudanpur,
        "A time delay neural network architecture for 
        efficient modeling of long temporal contexts",
        Proc. Interspeech 2015, 2015, pp. 3214-3218.

        [2] D. Snyder, D. Garcia-Romero, G. Sell, 
        D. Povey and S. Khudanpur, "X-Vectors: Robust DNN 
        Embeddings for Speaker Recognition", 2018 IEEE 
        International Conference on Acoustics, Speech and 
        Signal Processing (ICASSP), 2018, pp. 5329-5333.
    
    """
    def __init__(
        self, 
        embeddings_dim: int = 512,
        num_features: int = 30,
        num_classes: int = 7185
    ) -> None:
        super(TDNN, self).__init__()

        self.embeddings = nn.Sequential(
            TDNNLayer(context_size=5, in_channels=num_features),
            DenseLayer(in_features=512, out_features=512),
            TDNNLayer(context_size=3, dilation=2),
            DenseLayer(in_features=512, out_features=512),
            TDNNLayer(context_size=3, dilation=3),
            DenseLayer(in_features=512, out_features=512),
            TDNNLayer(context_size=3, dilation=4),
            DenseLayer(in_features=512, out_features=512),
            DenseLayer(in_features=512, out_features=512),
            DenseLayer(in_features=512, out_features=1500),
            StatsPoolingLayer(),
            nn.Linear(in_features=3000, out_features=embeddings_dim)
        )

        self.output_seq = nn.Sequential(
            nn.ReLU(),
            DenseLayer(in_features=embeddings_dim, out_features=512),
            nn.Linear(in_features=512, out_features=512),
            nn.Softmax(dim=num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode="fan_out", 
                    nonlinearity="relu"
                )
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x)
        if self.training:
            x = self.output_seq(x)
        return x


def conv1x1(
    in_channels: int, 
    out_channels: int, 
    stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )


def conv3x3(
    in_channels: int, 
    out_channels: int, 
    stride: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False
    )


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(ResNetBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(SpeakerRecognitionModel):
    """ResNet34 model, as described in [1]. The
    implementation is a simplified and slightly
    modified version of the official PyTorch 
    Vision ResNet.

    Before the final fully-connected layer,
    we added a statistics pooling layer, as
    described in [2].

    References
    ----------
        [1] K. He, X. Zhang, S. Ren and J. Sun, "Deep 
        Residual Learning for Image Recognition", 2016 IEEE 
        Conference on Computer Vision and Pattern Recognition 
        (CVPR), 2016, pp. 770-778.

        [2] D. Snyder, D. Garcia-Romero, G. Sell, 
        D. Povey and S. Khudanpur, "X-Vectors: Robust DNN 
        Embeddings for Speaker Recognition", 2018 IEEE 
        International Conference on Acoustics, Speech and 
        Signal Processing (ICASSP), 2018, pp. 5329-5333.
    """
    def __init__(self, **kwargs) -> None:
        super(ResNet34, self).__init__(**kwargs)

        self.current_channels = 64

        self.conv1 = nn.Conv2d(
            1, 
            self.current_channels, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.current_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_sequence(64, num_blocks=3)
        self.conv3_x = self._make_sequence(128, num_blocks=4, stride=2)
        self.conv4_x = self._make_sequence(256, num_blocks=6, stride=2)
        self.conv5_x = self._make_sequence(512, num_blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sp = StatsPoolingLayer()
        self.fc = nn.Linear(512, self.embeddings_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode="fan_out", 
                    nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.sp(x)
        x = self.fc(x)

        return x
    
    def _make_sequence(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ):
        downsample = None

        # downsample when we increase the dimension, using
        # the 1x1 convolution option, as described in the
        # ResNet paper.
        if stride != 1 or self.current_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.current_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            ResNetBlock(
                in_channels=self.current_channels, 
                out_channels=out_channels, 
                stride=stride, 
                downsample=downsample
            )
        )
        self.current_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                ResNetBlock(
                    in_channels=self.current_channels,
                    out_channels=out_channels
                )
            )

        return nn.Sequential(*layers)