import math
import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxLoss(nn.Module):
    def __init__(
        self, 
        embeddings_dim: int = 512, 
        num_classes: int = 5
    ) -> None:
        super(SoftmaxLoss, self).__init__()
        self.clf = nn.Linear(embeddings_dim, num_classes)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        logits = self.clf(x)
        loss = self.ce(logits, label)
        return loss, logits


class ArcFaceLoss(nn.Module):
    """Implementation of ArcFace Loss, also called
    Additive Angular Margin Softmax (AAM Softmax), as
    described in [1].

    This implementation is adapted from the one
    provided by the authors of ArcFace at:
        https://github.com/deepinsight/insightface

    References
    ----------
        [1] J. Deng et al., "ArcFace: Additive Angular 
        Margin Loss for Deep Face Recognition", 2019,
        https://arxiv.org/abs/1801.07698.
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        
        # cos(target+margin)
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale

        loss = self.ce(logits, labels)
        return loss, logits


class AAMSoftmaxLoss(nn.Module):
    """Implementation of ArcFace Loss, also called
    Additive Angular Margin Softmax (AAM Softmax), as
    described in [1].

    This code is adapted from the one
    provided as baseline for the VoxCeleb Speaker 
    Recognition Challenge (VoxSRC):
        https://github.com/clovaai/voxceleb_trainer

    References
    ----------
        [1] J. Deng et al., "ArcFace: Additive Angular 
        Margin Loss for Deep Face Recognition", 2019,
        https://arxiv.org/abs/1801.07698.
    """

    def __init__(
        self, 
        embeddings_dim, 
        num_classes, 
        margin=0.5, 
        scale=64, 
        easy_margin=False 
    ):
        super(AAMSoftmaxLoss, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = embeddings_dim
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(num_classes, embeddings_dim), 
            requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing 
        # while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta). This is basically W^t*X
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # sqrt(1 - cos^2(theta)) = sin(theta) for x in quadrants 1 or 2
        # otherwise it's -sqrt...
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # cos(theta + m) 
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # Here we have only the exponent of the exp, then
        # we apply CrossEntropy to get the log of the SoftMax
        logits = logits * self.s

        loss = self.ce(logits, label)
        return loss, logits


class SubCenterAAMSoftmaxLoss(nn.Module):
    """Implementation of Sub-center ArcFace Loss, also 
    called Sub-center AAM Softmax, as described in [1].
    The implementation is based on the images of page
    6 of the paper.

    References
    ----------
        [1] J. Deng et al., "Sub-center ArcFace: Boosting 
        Face Recognition by Large-Scale Noisy Web Faces", 
        Computer Vision - ECCV 2020, 2020, pp. 741-757.
    """

    def __init__(
        self, 
        embeddings_dim, 
        num_classes,
        num_subcenters=3,
        margin=0.5, 
        scale=64,
        easy_margin=False
    ):
        super(SubCenterAAMSoftmaxLoss, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.num_subcenters = num_subcenters
        self.in_feats = embeddings_dim
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(
                num_classes, 
                num_subcenters, 
                embeddings_dim
            ), 
            requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing 
        # while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        ls = []

        for batch in x:
            batch = F.normalize(batch, dim=-1)
            ls.append(
                torch.matmul(F.normalize(self.weight, dim=-1), batch)
            )

        subclass_cosine = torch.stack(ls)
        max_pool = F.max_pool1d(subclass_cosine, self.num_subcenters)
        theta = torch.arccos(max_pool).squeeze(-1)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where((cos_theta - self.th) > 0, phi, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits = logits * self.s
        
        loss = self.ce(logits, label)
        return loss, logits
