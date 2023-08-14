import torch
import torch.nn as nn


# 基于负对数似然的loss
# pose表示w2c
# Tx,Ty,Tz基于欧几里得距离简单建模分布
# quaternion表示的Rotation基于custom 距离度量建立单维的分布: arccos((tr(MM_{GT}^T)-1)/2)
class CoordiNetLoss(nn.Module):
    def __init__(self, loss_type='homosc', learn_beta=True, c=1):
        super().__init__()
        self.loss_type = loss_type
        self.c = c  # R loss的系数
        self.learn_beta = learn_beta
        beta = torch.zeros(4, dtype=torch.float32)
        self.beta = nn.Parameter(beta, requires_grad=learn_beta)

    def homosc(self, t, q, v, y_pose):
        """
        Homosc loss, 不考虑uncertainty,包括:
        1. 可学习的geometric loss即self.beta; 2. 不可学习的基本loss即t用Euler dist度量, R用arccos(tr(MM_{GT}^T)-1)/2度量
        :param t: (n,3):[Tx,Ty,Tz]
        :param q: (n,4):[w,x,y,z]
        :param v: (n,4):[var(Tx),var(Ty),var(Tz),var(R)]
        :param y_pose: (n,3,4): ground truth of w2c rotation matrix and c2w t vector(cam center)
        :return:
        """
        R = self.batch_quat2rotmat(q)  # 批量四元数->rotation mat
        y_R, y_t = y_pose[:, :, :3], y_pose[:, :, -1]  # groundTruth pose分解为R和T
        # 1. t vector loss
        loss_t = torch.mean((t - y_t) ** 2, dim=0)
        # 2. R mat loss
        similarity = (torch.sum(torch.diagonal(torch.bmm(R, torch.transpose(y_R, -1, -2)), dim1=-2, dim2=-1),
                                dim=1) - 1) / 2
        angle_diff = torch.acos(torch.clamp(similarity, -1, 1))
        t_diff = torch.norm((t - y_t), dim=1)
        loss_R = torch.mean(angle_diff)
        if self.learn_beta:
            loss = (loss_t * torch.exp(-self.beta[:3])).sum() + loss_R * torch.exp(-self.beta[-1]+3) + self.beta.sum()
        else:
            loss = torch.sum(loss_t) + loss_R * self.c
        return loss, loss_t.mean().item(), loss_R.item(), t_diff, angle_diff

    def heterosc(self, t, q, v, y_pose):
        """
        Heterosc loss,注意:
        1. t vector loss计算三个方向分量独立即不除以3
        2. R loss 的负对数似然的距离度量非简单欧几里得且可视为将R建模为单变量的分布, R只伴随输出一个variance。
        3. 传统的负对数似然乘2。
        :param t: (n,3):[Tx,Ty,Tz]
        :param q: (n,4):[w,x,y,z]
        :param v: (n,4):[var(Tx),var(Ty),var(Tz),var(R)]
        :param y_pose: (n,3,4): ground truth of rotation matrix and t vector
        :return:
        """
        n = len(t)
        R = self.batch_quat2rotmat(q)  # 批量四元数->rotation mat
        v_t, v_R = v[:, :3], v[:, -1]  # t、R的variance
        y_R, y_t = y_pose[:, :, :3], y_pose[:, :, -1]  # groundTruth pose分解为R和T
        # 1. t vector loss
        loss_t = torch.sum((t - y_t) ** 2 / v_t) / n + torch.sum(torch.log(v_t)) / n
        # 2. R mat loss
        similarity = (torch.sum(torch.diagonal(torch.bmm(R, torch.transpose(y_R, -1, -2)), dim1=-2, dim2=-1),
                                dim=1) - 1) / 2
        t_diff = torch.norm((t - y_t), dim=1)
        angle_diff = torch.acos(torch.clamp(similarity, -1, 1))
        loss_R = torch.mean(angle_diff / v_R) + torch.mean(torch.log(v_R))
        loss = loss_t + loss_R
        return loss, loss_t.item(), loss_R.item(), t_diff, angle_diff

    def forward(self, t, q, v, y_pose):
        if self.loss_type == 'homosc':
            return self.homosc(t, q, v, y_pose)
        elif self.loss_type == 'heterosc':
            return self.heterosc(t, q, v, y_pose)

    def batch_quat2rotmat(self, q):
        """
        批量quat2rotmat: torch.tensor类型
        :param q: (n,4)
        :return:
        """
        w, x, y, z = torch.split(q, [1, 1, 1, 1], dim=1)
        R = torch.hstack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
                          2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w,
                          2 * x * z - 2 * w * y, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2])
        return R.view(-1, 3, 3)
