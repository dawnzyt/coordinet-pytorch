import torch
import torch.nn as nn

import option.coordinet_option
from model.coordinet_basemodel import CoordiNet
from efficientnet_pytorch import EfficientNet
from loss.coordinet_loss import CoordiNetLoss
from torchvision import models


class CoordiNetSystem(nn.Module):

    def __init__(self, input_dim=1536, W=256, loss_type='homosc', learn_beta=True, var_min=[0.01, 0.01, 0.01, 0.01],
                 fixed_weight=False,backbone='efficientnet', backbone_path=None):
        super().__init__()
        self.loss_type = loss_type
        self.var_min = var_min
        self.backbone=backbone
        self.coordinet = CoordiNet(input_dim, W)
        # resnet34
        if backbone=='resnet':
            base_model=models.resnet34(pretrained=True)
            self.encoder=nn.Sequential(*list(base_model.children())[:-2])
            print('load pretrained resnet34 model')
        # efficientnet-b3
        elif backbone=='efficientnet':
            if backbone_path is None:
                self.encoder = EfficientNet.from_pretrained('efficientnet-b3')
            else:
                self.encoder = EfficientNet.from_pretrained('efficientnet-b3', weights_path=backbone_path)
        # 冻结参数
        if fixed_weight:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if self.loss_type == 'homosc':
            for param in self.coordinet.variance_decoder.parameters():
                param.requires_grad = False
            for param in self.coordinet.variance_encoder.parameters():
                param.requires_grad = False
        self.loss_fn = CoordiNetLoss(loss_type, learn_beta)

    def forward(self, x, cal_loss=False, y_pose=None):
        """

        :param
        x: (B, 3, H, W)
        批量images
        :param
        cal_loss: 是否计算loss
        :param
        y_pose: (B, 3, 4)
        批量ground
        truth
        :return: t, q, v, losses
        """
        if self.backbone=='resnet':
            feats = self.encoder(x)
        else:
            feats = self.encoder.extract_features(x)
        t, q, v = self.coordinet(feats)
        v[:, 0] += self.var_min[0]  # bias
        v[:, 1] += self.var_min[1]
        v[:, 2] += self.var_min[2]
        v[:, 3] += self.var_min[3]
        losses = None
        if cal_loss:
            losses = dict()
            loss, loss_t, loss_R, t_diff, angle_diff = self.loss_fn(t, q, v, y_pose)
            losses['loss'] = loss
            losses['t'] = loss_t
            losses['R'] = loss_R
            losses['t_diff'] = t_diff
            losses['angle_diff'] = angle_diff
        return t, q, v, losses
# opt=option.coordinet_option.CoordiNetOption().into_opt()
# base_model=EfficientNet.from_pretrained('efficientnet-b3', weights_path=opt.backbone_path)
# print(base_model)
# base_model=nn.Sequential(*list(base_model.children()))
# img=torch.randn(1,3,224,224)
# x=base_model(img)
# print(x.shape)

# base_model=models.resnet34(pretrained=True)
# model=nn.Sequential(*list(base_model.children())[:-2])
# print(base_model)
# img=torch.randn(1,3,224,224)
# x=model(img)
# print(x.shape)
