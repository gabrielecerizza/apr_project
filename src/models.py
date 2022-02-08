from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import Accuracy, F1Score

from src.losses import ArcFaceLoss, AAMSoftmaxLoss

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


class ResNet34(LightningModule):
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
    def __init__(
        self,
        num_classes: int,
        embeddings_dim: int = 512,
        loss_margin: float = 0.5,
        loss_scale: float = 64,
        average: str = "weighted"
    ) -> None:
        super(ResNet34, self).__init__()

        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.current_channels = 64
        self.loss_func = AAMSoftmaxLoss(
            num_classes=num_classes,
            embeddings_dim=embeddings_dim,
            margin=loss_margin,
            scale=loss_scale
        )

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
        self.fc = nn.Linear(512, embeddings_dim)

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

        self.acc = Accuracy(
            num_classes=num_classes,
            average=average
        )

        self.f1 = F1Score(
            num_classes=num_classes,
            average=average
        )

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

    def training_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        x = x.unsqueeze(1)
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)

        train_acc = self.acc(logits, true_labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        x = x.unsqueeze(1)
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)

        val_acc = self.acc(logits, true_labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, true_labels = batch["features"], batch["labels"]
        x = x.unsqueeze(1)
        out = self(x)
        loss, logits = self.loss_func(out, true_labels)
        return loss

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