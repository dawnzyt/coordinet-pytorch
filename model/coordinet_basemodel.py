import torch
import torch.nn as nn
import torch.nn.functional as F


# backbone: Efficient-Net B3
# https://readpaper.com/pdf-annotate/note?pdfId=4545272359715233793&noteId=1898796528097870592

# CoordConv layer
class CoordConv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding,bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        1. add i、j coordinates
        2. self.conv forward
        :param x: (B,C,H,W)
        :return:
        """
        b, c, h, w = x.shape
        i_range = torch.linspace(-1, 1, w, device=x.device)  # x坐标
        j_range = torch.linspace(-1, 1, h, device=x.device)  # y坐标
        j, i = torch.meshgrid(j_range, i_range)  # (h,w) meshgrid
        coord_cat = torch.stack([i, j], dim=0)  # (2,h,w)
        # return (B,C+2,H,W)
        x = torch.cat([x, torch.unsqueeze(coord_cat, 0).expand(b, -1, -1, -1)], dim=1)
        return self.conv(x)


class ConfidenceWeighedAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Softplus()

    def forward(self, x):
        """
        默认最后一个通道作为weight
        :param x: (B,C,H,W)
        :return: y:(B,C-1,1,1)
        """
        weight = self.activation(x[:, -1:, ...])
        # weight = weight / torch.sum(weight, dim=[-2, -1], keepdim=True)
        return torch.sum(x[:, :-1, ...] * weight, dim=[-2, -1])


# CoordiNet: 结构直接完全复现
class CoordiNet(nn.Module):
    def __init__(self, input_dim=1536, W=256):
        super().__init__()
        # 1. encode, decode pose
        self.pose_encoder_front = nn.Sequential(CoordConv(input_dim, W, (1, 1)) ,nn.ReLU(True),
                                                CoordConv(W, W, (3, 3), padding=1),nn.ReLU(True),
                                                CoordConv(W, W, (3, 3), padding=1), nn.ReLU(True))
        self.tvec_decode = nn.Sequential(CoordConv(W, 4, (1, 1)), ConfidenceWeighedAvgPool2d())
        self.qvec_decode = nn.Sequential(CoordConv(W, 4, (1, 1)), nn.AdaptiveAvgPool2d(output_size=1))
        # 2. encode, decode variance
        self.variance_encoder = nn.Sequential(nn.Conv2d(input_dim, W, (1, 1)), nn.ReLU(True),
                                              nn.Conv2d(W, W, (3, 3), padding=1),  nn.ReLU(True),
                                              nn.Conv2d(W, W, (3, 3), padding=1) , nn.ReLU(True),
                                              nn.Conv2d(W, 4, (1, 1)))
        for layer in self.variance_encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.variance_decoder = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1), nn.ReLU())

    def forward(self, x):
        """

        :param x: (B,C,H,W)
        :return:
        """
        pose_encoding = self.pose_encoder_front(x)
        t_vec = self.tvec_decode(pose_encoding)
        q_vec = self.qvec_decode(pose_encoding)
        var = self.variance_decoder(self.variance_encoder(x))
        return t_vec, self.normalize(q_vec.view(q_vec.shape[:2])), var.view(var.shape[:2])

    def normalize(self, x, eps=1e-12):
        """
        这里只用于标准化四元数
        Input: x:(n,4)
        :return:
        """
        return x / (torch.norm(x, dim=1, keepdim=True) + eps)