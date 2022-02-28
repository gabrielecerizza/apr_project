from asyncio import staggered
from itertools import cycle
import json
import math
import os
from typing import Callable, Optional, Union

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
        optimizer: Union[Callable, str] = "AdamW",
        lr_scheduler: Union[Callable, str] = None,
        lr_scheduler_interval: str = "step",
        average: str = "weighted",
        num_secs: int = 3,
        base_path: str = "E:/Datasets/VoxCeleb1/subset/",
        feature_type: str = "logmel",
        embeddings_strategy: str = "separate",
        cohort_strategy: str = "separate",
        normalization_strategy: str = "s_norm",
        top_n: int = 100
    ) -> None:
        super(SpeakerRecognitionModel, self).__init__()

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.average = average
        self.num_secs = num_secs
        self.base_path = base_path
        self.feature_type = feature_type
        self.embeddings_strategy = embeddings_strategy
        self.cohort_strategy = cohort_strategy
        self.normalization_strategy = normalization_strategy
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

        # self._set_optimizers()

    def _set_optimizers(self):
        if self.optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=0.256,
                momentum=0.9,
                weight_decay=0.99
            )
        elif self.optimizer == "Adam":
            self.optimizer = torch.optim.NAdam(
                self.parameters(),
                lr=0.001,
                weight_decay=0
            )
        elif self.optimizer == "NAdam":
            self.optimizer = torch.optim.NAdam(
                self.parameters()
            )
        elif self.optimizer is None or self.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=1e-3, 
                eps=1e-8
            )
        if self.lr_scheduler == "StepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                gamma=0.97,
                step_size=2
            )
        elif self.lr_scheduler == "CyclicLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer=self.optimizer,
                base_lr=1e-5,
                max_lr=1e-2,
                step_size_up=5000,
                cycle_momentum=False,
                mode="triangular2"
            )
        elif self.lr_scheduler == "ReducePlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=2
            )
        

    def set_optimizer(
        self, 
        optimizer, 
        lr_scheduler, 
        lr_scheduler_interval
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval

    def create_embeddings(self):
        embeddings_ls = []

        df = pd.read_csv(
            self.base_path + f"subset_features_{self.num_secs}.csv"
        )
        df = df[df["Type"] == self.feature_type]
        for index, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Creating embeddings"
        ):
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

    def compute_scores(
        self, 
        batch, 
        cohort_strategy="separate",
        normalization_strategy="s_norm"
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
            self.base_path + f"subset_embeddings_{self.num_secs}.csv"
        )

        speaker_embeddings_dict = dict()

        for index, row in df.iterrows():
            if row["Set"] == "train":
                speaker = row["Speaker"]
                embedding_filename = row["File"]
                embedding = torch.load(embedding_filename)
                speaker_embeddings_dict.setdefault(speaker, []).append(embedding) 

        speakers = list(speaker_embeddings_dict.keys())
        
        if cohort_strategy == "mean":
            cohort = torch.vstack(
                [
                    torch.mean(
                        torch.vstack(speaker_embeddings_dict[speaker]), 
                        axis=0,
                        keepdims=True
                    ) 
                    for speaker in speakers
                ]
            )
        elif cohort_strategy == "separate":
            cohort = torch.vstack(
                [   
                    torch.vstack(speaker_embeddings_dict[speaker])
                    for speaker in speakers
                ]
            )
        else:
            raise ValueError("unknown strategy in compute_scores")

        for speaker, embeddings_ls in tqdm(
            speaker_embeddings_dict.items(),
            total=len(speaker_embeddings_dict),
            desc="Computing scores"
        ):
            for embedding in embeddings_ls:
                e_distances = torch_cosine_distances(
                    embedding.unsqueeze(0), cohort
                )[0]
                e_distances, indices = torch.sort(e_distances)
                e_distances = e_distances[:self.top_n]

                me = torch.mean(e_distances)
                se = torch.std(e_distances, unbiased=True)

                for idx, test_speaker in enumerate(batch["speakers"]):
                    test_embedding = batch["embeddings"][idx]

                    if normalization_strategy is None:
                        # print(embedding.shape, test_embedding.shape)
                        score = F.cosine_similarity(
                            embedding, test_embedding, dim=0
                        )

                    elif normalization_strategy == "s_norm":
                        distance = torch_cosine_distances(
                            embedding.unsqueeze(0), test_embedding.unsqueeze(0)
                        )[0]

                        t_distances = torch_cosine_distances(
                            test_embedding.unsqueeze(0), cohort
                        )[0]
                        t_distances, indices = torch.sort(t_distances)
                        t_distances = t_distances[:self.top_n]

                        mt = torch.mean(t_distances)
                        st = torch.std(t_distances, unbiased=True)

                        e_term = (distance - me) / (se + 1) # + 1 to avoid 0
                        t_term = (distance - mt) / (st + 1)
                        
                        # We negate the score so that the score 
                        # is higher if the embeddings are similar
                        score = -0.5 * (e_term + t_term)

                    else:
                        raise ValueError(
                            "unknown normalization "
                            "argument in compute_scores"
                        )

                    if score.isnan():
                        print("NaN score:", speaker, idx, test_speaker, "\n")
                        print("e_term", e_term, "\n")
                        print("t_term", t_term, "\n")
                        print("distance", distance, "\n")
                        print("me se", me, se, "\n")
                        print("st st", mt, st, "\n")
        
                    scores.append(score)
                    labels.append(int(speaker == test_speaker))

        scores = torch.tensor(scores)
        labels = torch.tensor(labels)
        assert not scores.isnan().any()
        assert not labels.isnan().any()

        return scores, labels 

    def training_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)

        train_acc = self.acc(logits, true_labels)
        train_f1 = self.f1(logits, true_labels)

        metrics_ls = [
            ("train_loss", loss),
            ("train_acc", train_acc), 
            ("train_f1", train_f1)
        ]
        for metric_name, metric_val in metrics_ls:
            self.log(
                metric_name, 
                metric_val,
                prog_bar=True,
                on_step=True, 
                on_epoch=True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)
        batch["embeddings"] = out

        val_acc = self.acc(logits, true_labels)
        val_f1 = self.f1(logits, true_labels)

        """
        scores, labels = self.compute_scores(
            batch,
            cohort_strategy=self.cohort_strategy,
            normalization_strategy=self.normalization_strategy
        )
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        val_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        val_eer = compute_eer(scores.numpy(), labels.numpy())
        """

        metrics_ls = [
            ("val_loss", loss),
            ("val_acc", val_acc), 
            ("val_f1", val_f1), 
            # ("val_min_dcf", val_min_dcf),
            # ("val_eer", val_eer)
        ]
        for metric_name, metric_val in metrics_ls:
            self.log(
                metric_name, 
                metric_val,
                prog_bar=True,
                on_step=True, 
                on_epoch=True
            )

        return loss

    def on_test_epoch_start(self) -> None:
        self.create_embeddings()

        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)
        batch["embeddings"] = out

        test_acc = self.acc(logits, true_labels)
        test_f1 = self.f1(logits, true_labels)

        scores, labels = self.compute_scores(
            batch,
            cohort_strategy=self.cohort_strategy,
            normalization_strategy=self.normalization_strategy
        )
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        test_min_dcf, _ = compute_min_dcf(
            fnrs=fnrs, 
            fprs=fprs, 
            thresholds=thresholds
        )
        test_eer = compute_eer(scores.numpy(), labels.numpy())

        metrics_ls = [
            ("test_acc", test_acc),
            ("test_f1", test_f1), 
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
            "model_name": model_name,
            "acc": float(self.acc.compute().cpu().numpy()),
            "f1": float(self.f1.compute().cpu().numpy()),
            "min_dcf": float(min_dcf),
            "eer": float(eer),
            "embeddings_dim": self.embeddings_dim,
            "loss": str(self.loss_func),
            "optimizer": str(self.optimizer),
            "lr_scheduler": str(self.lr_scheduler),
            "lr_scheduler_interval": self.lr_scheduler_interval,
            "average": self.average,
            "num_secs": self.num_secs,
            "feature_type": self.feature_type,
            "embeddings_strategy": self.embeddings_strategy,
            "cohort_strategy": self.cohort_strategy,
            "normalization_strategy": self.normalization_strategy,
            "top_n": self.top_n
        }

        with open(
            f"{save_dir}/{model_name}_epoch={self.current_epoch}.json", 
            "w", encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)

    def configure_optimizers(self):
        res = dict()
        res["optimizer"] = self.optimizer

        if self.lr_scheduler is not None:
            res["lr_scheduler"] = {
                "scheduler": self.lr_scheduler,
                "interval": self.lr_scheduler_interval,
                "monitor": "val_loss"
            }
        return res

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
        Proc. Interspeech 2015, 2015.
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


class SelfAttentionPooling(nn.Module):
    """Implementation of Self Attention Pooling (SAP) as
    described in [1]. We used GELU instead of tanh as
    non-linearity.

    References
    ----------
        [1] W. Cai, J. Chen and M. Li, "Exploring the Encoding 
        Layer and Loss Function in End-to-End Speaker and 
        Language Recognition System", 2018,
        https://arxiv.org/abs/1804.05160
    """
    def __init__(
        self,
        n_mels
    ) -> None:
        super(SelfAttentionPooling, self).__init__()
        self.linear = nn.Linear(n_mels, n_mels)
        self.attention = nn.Parameter(
            torch.FloatTensor(size=(n_mels, 1))
        )
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax2d()

        nn.init.xavier_normal_(self.attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0,3,1,2)
        h = self.gelu(self.linear(y))
        mul = torch.matmul(h, self.attention)
        w = self.softmax(mul)
        w = w.permute(0,2,3,1)
        e = torch.sum(x * w, dim=-1)
        
        return e


class VarSelfAttentionPooling(nn.Module):
    """Variant of self-attention pooling to further
    decrease the number of dimensions. We perform
    a weighted sum of the n_mels this time, so we
    can remove the last dimension.
    """
    def __init__(
        self,
        n_channels,
        n_mels
    ) -> None:
        super(VarSelfAttentionPooling, self).__init__()
        self.linear = nn.Linear(n_channels, n_channels)
        self.attention = nn.Parameter(
            torch.FloatTensor(size=(n_channels, n_mels))
        )
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax2d()

        nn.init.xavier_normal_(self.attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0,2,1)
        h = self.gelu(self.linear(y))
        mul = torch.matmul(h, self.attention)
        w = self.softmax(mul)
        e = torch.sum(x.bmm(w), dim=-1)
        
        return e


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
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = SelfAttentionPooling(3)
        self.pool2 = VarSelfAttentionPooling(512, 3)
        # self.sp = StatsPoolingLayer()
        self.embeddings = nn.Linear(512, self.embeddings_dim)
        # self.clf = nn.Linear(self.embeddings_dim, self.num_classes)

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

        self._set_optimizers()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.pool1(x)
        x = self.pool2(x)
        # x = self.pool(x)
        # x = torch.flatten(x, 1)
        # x = self.sp(x)
        x = self.embeddings(x)

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
        [1] J. Hu et al., "Squeeze-and-Excitation Networks", 
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
        Based Speaker Verification", Proc. Interspeech 
        2020, 2020, pp. 3830-3834.

        [2] S.-H. Gao et al., "Res2Net: A New Multi-Scale 
        Backbone Architecture", IEEE Transactions on Pattern 
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


class Var_SERes2Block(SERes2Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out_ls = torch.split(out, out.size(1)//self.scale, dim=1)
        y_ls = []
        
        for idx in range(self.scale):
            out_split = out_ls[idx]
            k_fun = self.K[idx]
            y_ls.append(k_fun(out_split))

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
        Based Speaker Verification", Proc. Interspeech 
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
        Based Speaker Verification", Proc. Interspeech 
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
        self.se1 = Var_SERes2Block(n_channels, scale, dilation=2)
        self.se2 = Var_SERes2Block(n_channels, scale, dilation=3)
        self.se3 = Var_SERes2Block(n_channels, scale, dilation=4)
        self.conv2 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1, padding=1)
        self.attn_pool = AttentiveStatPooling(n_channels, n_channels // 10, conv=True)
        self.bn2 = nn.BatchNorm1d(n_channels * 2)
        self.embeddings = nn.Linear(n_channels * 2, self.embeddings_dim)
        self.clf = nn.Linear(self.embeddings_dim, self.num_classes)

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
        out = self.embeddings(out)

        # if self.training:
        #    out = self.clf(out)
    
        return out


class ResLSTM(nn.Module):
    """Variant of a residual block with bidirectional LSTM
    described in [1].

    References
    ----------
        [1] Y. Zhang, W. Chan and N. Jaitly, "Very Deep 
        Convolutional Networks for End-to-End Speech 
        Recognition", 2016, https://arxiv.org/abs/1610.03022
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


class MHA_LAS(SpeakerRecognitionModel):
    """Variant of the LAS model described in [1], with
    the encoder being extended according to the 
    Transformer architecture [2].

    References
    ----------
        [1] K. Irie et al., "On the Choice of Modeling 
        Unit for Sequence-to-Sequence Speech Recognition",
        Interspeech 2019, 2019.

        [2] A. Vaswani et al., "Attention Is All You Need,"
        2017, https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self, 
        n_channels: int = 32,
        n_mels: int = 80,
        num_heads: int = 8,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super(MHA_LAS, self).__init__(**kwargs)

        self.n_channels = n_channels
        self.num_heads = num_heads
        self.head_dim = self.embeddings_dim // num_heads

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
        self.embeddings = nn.Sequential(
            nn.Linear(n_channels * n_mels, self.embeddings_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(self.embeddings_dim // 2, self.embeddings_dim)
        )

        self.clf = nn.Linear(self.embeddings_dim, self.num_classes)

        self._initialize_weights()
        self._set_optimizers()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    
        out = self.embeddings(out)
        # if self.training:
        #    out = self.clf(out)

        return out


class ConvSEBlock(nn.Module):
    """Squeeze and excitation block using convolution."""
    def __init__(
        self,
        n_channels: int,
        reduction_ratio: int = 4 # 0.25 se_ratio in original
    ) -> None:
        super(ConvSEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(
            n_channels, 
            n_channels // reduction_ratio,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True
        )
        self.gelu = nn.GELU()
        self.se_expand = nn.Conv2d(
            n_channels // reduction_ratio, 
            n_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(x)
        out = self.se_reduce(out)
        out = self.gelu(out)
        out = self.se_expand(out)
        out = self.sigmoid(out)
        return out * x


class MBConvBlock(nn.Module):
    """MBConv and Fused-MBConv block, as described in [1]
    and [2].

    References
    ----------
        [1] M. Tan and Q.V. Le, "EfficientNetV2: Smaller 
        Models and Faster Training", 2021,
        https://arxiv.org/abs/2104.00298 

        [2] M. Sandler et al., "MobileNetV2: Inverted Residuals 
        and Linear Bottlenecks", 2019, 
        https://arxiv.org/abs/1801.04381
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expand_ratio: int,
        stride: int,
        dropout_rate: float,
        se_ratio: Optional[int],
        fused: bool = False
    ) -> None:
        super(MBConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.se_ratio = se_ratio
        self.fused = fused
        self.hidden_dim = in_channels * expand_ratio

        self.has_se = se_ratio is not None and \
            0 < se_ratio <= 1

        if fused and expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if stride > 1 else "same",
                    bias=False
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.GELU(),
            )
        else:
            if expand_ratio != 1:
                self.expand = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.hidden_dim,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                        bias=False
                    ),
                    nn.BatchNorm2d(self.hidden_dim),
                    nn.GELU()
                )
            self.depthwise = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if stride > 1 else "same",
                    groups=self.hidden_dim,
                    bias=False
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.GELU()
            )

        self.dropout = nn.Dropout(dropout_rate)
        if self.has_se:
            self.se = ConvSEBlock(
                n_channels=self.hidden_dim,
                reduction_ratio=se_ratio
            )
        if fused:
            f_stride = 1 if expand_ratio != 1 else stride
            self.project_conv = nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=out_channels,
                kernel_size=1 if expand_ratio != 1 else kernel_size,
                stride=f_stride,
                padding=1 if f_stride > 1 else "same",
                bias=False
            )
            self.gelu = nn.GELU()
        else:
            self.project_conv = nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                bias=False
            )
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.expand_ratio != 1:
            out = self.expand(out)
        if not self.fused:
            out = self.depthwise(out)
        if self.expand_ratio > 1:
            out = self.dropout(out)
        if self.has_se:
            out = self.se(out)
        out = self.project_conv(out)
        out = self.norm2(out)

        if self.fused and self.expand_ratio == 1:
            out = self.gelu(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + x

        return out


def round_repeats(num_repeats, depth_coefficient):
    return int(math.ceil(num_repeats * depth_coefficient))


def round_channels(
    n_channels, 
    width_coefficient, 
    depth_divisor,
    min_depth = None
):
    if width_coefficient is None or width_coefficient == 1.0:
        return n_channels

    n_channels *= width_coefficient
    min_depth = min_depth or depth_divisor
    new_n_channels = max(
        min_depth,
        int(n_channels + depth_divisor / 2) // \
            depth_divisor * depth_divisor    
    )
    return int(new_n_channels)


class EfficientNetV2(SpeakerRecognitionModel):
    """Implementation of EfficientNetV2, as described 
    in [1]. The code is adapted from the official
    TensorFlow repository at:
        https://github.com/google/automl/blob/master/efficientnetv2

    References
    ----------
        [1] M. Tan and Q.V. Le, "EfficientNetV2: Smaller 
        Models and Faster Training", 2021,
        https://arxiv.org/abs/2104.00298 
    """
    def __init__(
        self,
        config: dict,
        stem_channels: int = 24,
        **kwargs
    ) -> None:
        super(EfficientNetV2, self).__init__(**kwargs)

        dropout_rate = config["dropout_rate"]
        self.stem_channels = stem_channels

        stem_channels = round_channels(
            n_channels=self.stem_channels,
            width_coefficient=config["width_coefficient"],
            depth_divisor=config["depth_coefficient"]
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(24),
            nn.GELU()
        )

        block_ls = []

        for block_config in config["blocks"]:
            block_args = block_config["args"]

            in_channels = round_channels(
                n_channels=block_args["in_channels"],
                width_coefficient=config["width_coefficient"],
                depth_divisor=config["depth_divisor"]
            )
            out_channels = round_channels(
                n_channels=block_args["out_channels"],
                width_coefficient=config["width_coefficient"],
                depth_divisor=config["depth_divisor"]
            )
            num_repeats = round_repeats(
                num_repeats=block_config["num_repeats"],
                depth_coefficient=config["depth_coefficient"]
            )
            block_args["in_channels"] = in_channels
            block_args["out_channels"] = out_channels
            block_config["num_repeats"] = num_repeats
            head_in = out_channels
    
            block = MBConvBlock(
                **block_args,
                dropout_rate=dropout_rate 
            )
            block_ls.append(block)

            if num_repeats > 1:
                block_args["in_channels"] = \
                    block_args["out_channels"]
                block_args["stride"] = 1
            
            for _ in range(num_repeats - 1):
                block = MBConvBlock(
                    **block_args,
                    dropout_rate=dropout_rate
                )
                block_ls.append(block)

        self.middle = nn.Sequential(
            *block_ls
        )

        head_out = round_channels(
            n_channels=1280,
            width_coefficient=config["width_coefficient"],
            depth_divisor=config["depth_divisor"]
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=head_in,
                out_channels=head_out,
                kernel_size=1,
                stride=1,
                padding="same",
                bias=False
            ),
            nn.BatchNorm2d(head_out),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate)
        )

        self.embeddings = nn.Linear(head_out, self.embeddings_dim)
        self.clf = nn.Linear(self.embeddings_dim, self.num_classes)

        self._initialize_weights()
        self._set_optimizers()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def summary(self) -> None:
        for m in self.modules():
            print(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        out = self.stem(x)
        out = self.middle(out)
        out = self.head(out)

        out = out.view(batch_size, 1280)
        out = self.embeddings(out)

        #if self.training:
        #    out = self.clf(out)
        
        return out


efficientnetv2_config = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "depth_divisor": 8,
    "dropout_rate": 0.1,
    "blocks": [
        {
            "num_repeats": 3,
            "args": {
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 1,
                "in_channels": 24,
                "out_channels": 24,
                "se_ratio": None,
                "fused": True
            } 
        },
        {
            "num_repeats": 5,
            "args": {
                "kernel_size": 3,
                "stride": 2,
                "expand_ratio": 4,
                "in_channels": 24,
                "out_channels": 48,
                "se_ratio": None,
                "fused": True
            } 
        },
        {
            "num_repeats": 5,
            "args": {
                "kernel_size": 3,
                "stride": 2,
                "expand_ratio": 4,
                "in_channels": 48,
                "out_channels": 80,
                "se_ratio": None,
                "fused": True
            } 
        },
        {
            "num_repeats": 7,
            "args": {
                "kernel_size": 3,
                "stride": 2,
                "expand_ratio": 4,
                "in_channels": 80,
                "out_channels": 160,
                "se_ratio": 4,
                "fused": False
            } 
        },
        {
            "num_repeats": 14,
            "args": {
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 6,
                "in_channels": 160,
                "out_channels": 176,
                "se_ratio": 4,
                "fused": False
            } 
        },
        {
            "num_repeats": 18,
            "args": {
                "kernel_size": 3,
                "stride": 2,
                "expand_ratio": 6,
                "in_channels": 176,
                "out_channels": 304,
                "se_ratio": 4,
                "fused": False
            } 
        },
        {
            "num_repeats": 5,
            "args": {
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 6,
                "in_channels": 304,
                "out_channels": 512,
                "se_ratio": 4,
                "fused": False
            } 
        }
    ]
    
}

def build_efficientnetv2(
    embeddings_dim, num_classes, loss_func,
    optimizer=None, lr_scheduler=None,
    lr_scheduler_interval=None
) -> EfficientNetV2:
    return EfficientNetV2(
        config=efficientnetv2_config,
        embeddings_dim=embeddings_dim,
        num_classes=num_classes,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_interval=lr_scheduler_interval
    )