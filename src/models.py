import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm

from src.metrics import (
    compute_eer, compute_error_rates, compute_min_dcf,
    torch_cosine_distances
)


class SpeakerRecognitionModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        embeddings_dim: int = 512,
        loss_func: Callable = None,
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
        self.average = average
        self.num_secs = num_secs
        self.base_path = base_path
        self.feature_type = feature_type
        self.embeddings_strategy = embeddings_strategy
        self.top_n = top_n
        self.loss_func = loss_func
            
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

            if row["Set"] != "train":
                continue

            file = row["File"]
            filename = os.path.splitext(os.path.basename(file))[0]
            features = torch.load(file).unsqueeze(1)
            if self.trainer.gpus >= 1:
                features = features.to("cuda")
            embeddings = self(features)[0]

            if self.embeddings_strategy == "mean":
                raise NotImplementedError
            elif self.embeddings_strategy == "separate":
                embeddings_file = self.base_path + "embeddings/" + row["Path"] \
                    + "/" + filename + "emb.pt"
                save_dir = os.path.dirname(embeddings_file)
                os.makedirs(save_dir, exist_ok=True)   
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
        cohort = torch.vstack(
            [
                torch.mean(
                    torch.vstack(speaker_embeddings[speaker]), 
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
            e_distances = torch_cosine_distances(
                embedding[0].unsqueeze(0), cohort
            )[0]
            e_distances, indices = torch.sort(e_distances)
            e_distances = e_distances[:self.top_n]

            me = torch.mean(e_distances)
            se = torch.std(e_distances)

            for idx, test_speaker in enumerate(batch["speakers"]):
                test_embedding = batch["embeddings"][idx]

                distance = torch_cosine_distances(
                    embedding[0].unsqueeze(0), test_embedding.unsqueeze(0)
                )[0]

                t_distances = torch_cosine_distances(
                    test_embedding.unsqueeze(0), cohort
                )[0]
                t_distances, indices = torch.sort(t_distances)
                t_distances = t_distances[:self.top_n]

                mt = torch.mean(t_distances)
                st = torch.std(t_distances)

                e_term = (distance - me) / se
                t_term = (distance - mt) / st
                score = 0.5 * (e_term + t_term)

                # We negate the score so that the score is
                # higher if the embeddings are similar
                scores.append(-score)
                labels.append(int(speaker == test_speaker))

        return torch.tensor(scores), torch.tensor(labels)

    def training_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
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
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)
        batch["embeddings"] = out

        val_acc = self.acc(logits, true_labels)

        scores, labels = self.compute_scores(batch)
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        val_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        val_eer = compute_eer(scores.numpy(), labels.numpy())

        metrics_ls = [
            ("val_loss", loss),
            ("val_acc", val_acc), 
            ("val_min_dcf", val_min_dcf),
            ("val_eer", val_eer)
        ]
        for metric_name, metric_val in metrics_ls:
            self.log(
                metric_name, 
                metric_val
            )

        return scores, labels

    def validation_epoch_end(self, outputs) -> None:
        model_name = self.__class__.__name__.lower()
        save_dir = "val_results/"
        os.makedirs(save_dir, exist_ok=True)

        scores, labels = [], []

        for tup in outputs:
            scores.extend(tup[0])
            labels.extend(tup[1])

        scores = torch.tensor(scores)
        labels = torch.tensor(labels)

        fnrs, fprs, thresholds = compute_error_rates(
            scores, labels
        )
        val_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        val_eer = compute_eer(scores.numpy(), labels.numpy())

        metrics_ls = [ 
            ("val_min_dcf", val_min_dcf),
            ("val_eer", val_eer)
        ]
        for metric_name, metric_val in metrics_ls:
            self.log(
                metric_name, 
                metric_val, 
                prog_bar=True,
                on_epoch=True
            )

        res = {
            "min_dcf": val_min_dcf.numpy().tolist(),
            "eer": val_eer,
            "scores": scores.numpy().tolist(),
            "labels": labels.numpy().tolist()
        }

        with open(
            f"{save_dir}/{model_name}_epoch={self.current_epoch}.json", 
            "w", encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)

        return super().validation_epoch_end(outputs)

    def on_test_epoch_start(self) -> None:
        self.create_embeddings()

        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        out = self(x)
        batch["embeddings"] = out

        scores, labels = self.compute_scores(batch)
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        test_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        test_eer = compute_eer(scores.numpy(), labels.numpy())

        metrics_ls = [ 
            ("test_min_dcf", test_min_dcf),
            ("test_eer", test_eer)
        ]
        for metric_name, metric_val in metrics_ls:
            self.log(
                metric_name, 
                metric_val, 
                prog_bar=True, 
                on_step=True, 
                on_epoch=True
            )

        return scores, labels

    def test_epoch_end(self, outputs) -> None:
        model_name = self.__class__.__name__.lower()
        save_dir = "results/"
        os.makedirs(save_dir, exist_ok=True)

        scores, labels = [], []

        for tup in outputs:
            scores.extend(tup[0])
            labels.extend(tup[1])

        scores = torch.tensor(scores)
        labels = torch.tensor(labels)

        fnrs, fprs, thresholds = compute_error_rates(
            scores, labels
        )
        min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        eer = compute_eer(scores.numpy(), labels.numpy())

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
            f"{save_dir}/{model_name}_epoch={self.current_epoch}.json", 
            "w", encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)

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

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) \
            if isinstance(limit_batches, int) \
            else int(limit_batches * batches)     

        num_devices = max(
            1, self.trainer.num_gpus, self.trainer.num_processes
        )
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


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
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
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
        return torch.stack([mean, std], dim=-1)


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
            nn.ReLU(inplace=True),
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
        self.relu = nn.ReLU()
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

    References
    ----------
        [1] K. He, X. Zhang, S. Ren and J. Sun, "Deep 
        Residual Learning for Image Recognition", 2016 IEEE 
        Conference on Computer Vision and Pattern Recognition 
        (CVPR), 2016, pp. 770-778.
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
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_sequence(64, num_blocks=3)
        self.conv3_x = self._make_sequence(128, num_blocks=4, stride=2)
        self.conv4_x = self._make_sequence(256, num_blocks=6, stride=2)
        self.conv5_x = self._make_sequence(512, num_blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.sp = StatsPoolingLayer()
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
        # x = self.sp(x)
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


class SEBlock(nn.Module):
    """Squeeze-and-excitation block, as described in [1].

    References
    ----------
        [1] J. Hu et al., "Squeeze-and-Excitation Networks," 
        IEEE Transactions on Pattern Analysis and Machine 
        Intelligence, vol. 42, no. 8, 2020, pp. 2011-2023.
    """
    def __init__(
        self, 
        n_channels: int,
        reduction_ratio: int = 16,
    ) -> None:
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.seq = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction_ratio),    
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction_ratio, n_channels),
            nn.Sigmoid()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, _, _ = x.size()
        out = self.pool(x).view(batch_size, n_channels)
        out = self.seq(out)
        return x * out.view(batch_size, n_channels, 1, 1)


class SERes2Block(nn.Module):
    """Variant of the SE-Res2Block described in [1]. We
    modified the architecture to follow more closely the
    Res2Block described in [2], by using 2d convolution.
    We also inverted the order of RELU and Batch 
    Normalization.

    References
    ----------
        [1] B. Desplanques et al., "ECAPA-TDNN: Emphasized 
        Channel Attention, Propagation and Aggregation TDNN 
        Based Speaker Verification," in Proc. Interspeech 
        2020, 2020, pp. 3830-3834.

        [2] S.-H. Gao et al., "Res2Net: A New Multi-Scale 
        Backbone Architecture," IEEE Transactions on Pattern 
        Analysis and Machine Intelligence, vol. 43, no. 2, 
        2021, pp. 652-662.
    """
    def __init__(
        self,
        n_channels: int,
        scale: int,
        dilation: int
    ) -> None:
        super(SERes2Block, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU()

        conv_ls = [
            nn.Conv2d(
                n_channels // scale, 
                n_channels // scale, 
                kernel_size=3,
                padding=dilation,
                dilation=dilation
            )
            for _ in range(scale - 1)
        ]

        self.K = nn.ModuleList([nn.Identity()] + conv_ls)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(n_channels)
        self.relu3 = nn.ReLU()
        self.se = SEBlock(n_channels)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out_ls = torch.split(out, out.size(1)//self.scale, dim=1)
        y_ls = []
        
        for idx in range(self.scale):
            out_split = out_ls[idx]
            k_fun = self.K[idx]
            if idx <= 1:
                y_ls.append(k_fun(out_split))
            else:
                prev = y_ls[idx - 1]
                y_ls.append(k_fun(out_split + prev))

        out = torch.cat(y_ls, dim=1)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.se(out)

        return out + x


class AttentiveStatPooling(nn.Module):
    """Attentive stat pooling layer, as described in [1].
    We provide also an implementation with convolution.
    Since the paper worked with MFCC instead of spectrograms,
    we averaged the values of mean and std, so that we
    could remove one dimension.

    References
    ----------
        [1] B. Desplanques et al., "ECAPA-TDNN: Emphasized 
        Channel Attention, Propagation and Aggregation TDNN 
        Based Speaker Verification," in Proc. Interspeech 
        2020, 2020, pp. 3830-3834. 
    """
    def __init__(
        self, 
        in_features: int, 
        latent_features: int,
        conv: bool = False
    ) -> None:
        super(AttentiveStatPooling, self).__init__()
        if conv:
            self.seq = nn.Sequential(
                nn.Conv2d(in_features, latent_features, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(latent_features, in_features, kernel_size=1)
            )
        else:
            self.seq = nn.Sequential(
                nn.Linear(in_features, latent_features),
                nn.ReLU(inplace=True),
                nn.Linear(latent_features, in_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = F.softmax(self.seq(x), dim=1)
        mean = torch.sum(attn_weights * x, dim=2).mean(-1)
        std = torch.sum(attn_weights * x ** 2, dim=2).mean(-1) - (mean ** 2)
        std = torch.sqrt(std)

        return torch.cat([mean, std], dim=1)


class Var_ECAPA(SpeakerRecognitionModel):
    """Variant of the ECAPA-TDNN model described in [1]. We
    omit the last batch normalization layer and adopt a
    different SE-Res2Block and Attentive Stats pooling
    layer. We also work on spectrograms instead of MFCC.

    References
    ----------
        [1] B. Desplanques et al., "ECAPA-TDNN: Emphasized 
        Channel Attention, Propagation and Aggregation TDNN 
        Based Speaker Verification," in Proc. Interspeech 
        2020, 2020, pp. 3830-3834. 
    """
    def __init__(
        self, 
        n_channels: int = 248,
        scale: int = 8, 
        **kwargs
    ) -> None:
        super(Var_ECAPA, self).__init__(**kwargs)

        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU()
        self.se1 = SERes2Block(n_channels, scale, dilation=2)
        self.se2 = SERes2Block(n_channels, scale, dilation=3)
        self.se3 = SERes2Block(n_channels, scale, dilation=4)
        self.conv2 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1, padding=1)
        self.attn_pool = AttentiveStatPooling(n_channels, n_channels // 10, conv=True)
        self.bn2 = nn.BatchNorm1d(n_channels * 2)
        self.fc = nn.Linear(n_channels * 2, self.embeddings_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out_se1 = self.se1(out)
        out_se2 = self.se2(out_se1)
        out_se3 = self.se3(out_se2)
        out = torch.cat([out_se1, out_se2, out_se3], dim=1)
        out = self.conv2(out)
        out = self.attn_pool(out)
        out = self.bn2(out)
        out = self.fc(out)
    
        return out


class ResLSTM(nn.Module):
    """Variant of a residual block with bidirectional LSTM
    described in [1].

    References
    ----------
        [1] Y. Zhang, W. Chan and N. Jaitly, "Very Deep 
        Convolutional Networks for End-to-End Speech 
        Recognition," 2016, https://arxiv.org/abs/1610.03022
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 512,
        target_size: int = 80,
        n_channels: int = 32
    ) -> None:
        super(ResLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size, target_size)
        self.bn = nn.BatchNorm1d(n_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = self.bn(out)
        out = self.gelu(out)

        return out + x


class MHA_LAS(nn.Module):
    """Variant of the LAS model described in [1], with
    the encoder being extended according to the 
    Transformer architecture [2].

    References
    ----------
        [1] K. Irie et al., "On the Choice of Modeling 
        Unit for Sequence-to-Sequence Speech Recognition,"
        Interspeech 2019, 2019.

        [2] A. Vaswani et al., "Attention Is All You Need,"
        2017, https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self, 
        embeddings_dim: int,
        n_channels: int = 32,
        n_mels: int = 80,
        num_heads: int = 8,
        dropout: float = 0.2
    ) -> None:
        super(MHA_LAS, self).__init__()

        self.embeddings_dim = embeddings_dim
        self.n_channels = n_channels
        self.num_heads = num_heads
        self.head_dim = embeddings_dim // num_heads

        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=3, stride=2)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.avg1 = nn.AdaptiveAvgPool2d((n_mels,1))
        self.reslstm1 = ResLSTM(n_mels)
        self.reslstm2 = ResLSTM(n_mels)
        self.reslstm3 = ResLSTM(n_mels)
        self.reslstm4 = ResLSTM(n_mels)

        self.qkv_proj = nn.Linear(n_mels, 3 * n_mels)
        self.mha = nn.MultiheadAttention(
            embed_dim=n_mels,
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(n_mels)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_mels * n_channels, self.embeddings_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(self.embeddings_dim // 2, self.embeddings_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.gelu1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu2(out)
        out = self.bn2(out)
        out = self.avg1(out).squeeze(-1)
        out = self.reslstm1(out)
        out = self.reslstm2(out)
        out = self.reslstm3(out)
        out = self.reslstm4(out)

        qkv = self.qkv_proj(out)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn, _ = self.mha(query=q, key=k, value=v)
        out = out + self.dropout(attn)
        out = self.norm1(out)

        batch_size, n_channels, n_mels = out.shape
        out = out.view(batch_size, n_channels * n_mels)
    
        out = self.mlp(out)

        return out