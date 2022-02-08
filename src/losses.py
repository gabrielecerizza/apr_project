import math
import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceLoss(torch.nn.Module):
    """Implementation of ArcFaceLoss, also called
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
    """Implementation of ArcFaceLoss, also called
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
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s

        loss = self.ce(logits, label)
        return loss, logits