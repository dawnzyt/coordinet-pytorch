import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, feat_dim, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.fc = nn.Linear(feat_dim, 2048, bias=True)
        self.fc_t = nn.Linear(2048, 3, bias=True)
        self.fc_q = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc, self.fc_t, self.fc_q]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, test_time=False):
        feat = self.fc(x)
        x = F.relu(feat)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=not test_time)

        t = self.fc_t(x)
        q = self.fc_q(x)
        q = F.normalize(q, p=2, dim=1)
        return t, q


# PoseNet
class PoseNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(PoseNet, self).__init__()
        self.dropout_rate = dropout_rate

        base_model = models.resnet34(pretrained=True)
        feat_dim = base_model.fc.in_features
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.decoder = Decoder(feat_dim, dropout_rate)

    def forward(self, x, test_time=False):
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        t, q = self.decoder(x, test_time)
        return t, q


class PoseLoss(nn.Module):

    def __init__(self):
        super(PoseLoss, self).__init__()
        self.sx = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
        self.sq = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)
        return

    def forward(self, t, q, y_t, y_q):
        loss_t = F.l1_loss(t, y_t)
        loss_q = F.l1_loss(q, y_q)
        loss = loss_t+math.exp(3) * loss_q - 3
        return loss,loss_t.item(), loss_q.item()
