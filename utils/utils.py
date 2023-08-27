import numpy as np
import cv2
import torch


# 四元数=>旋转矩阵
def quat2rotmat(q):
    """
    单位四元数=>旋转矩阵
    q:(w,x,y,z), 用单位四元数表示即w=cosθ
    """
    return np.array([
        [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
         2 * q[1] * q[2] - 2 * q[0] * q[3],
         2 * q[3] * q[1] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3],
         1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
         2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[3] * q[1] - 2 * q[0] * q[2],
         2 * q[2] * q[3] + 2 * q[0] * q[1],
         1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]])


def rotmat2quat(R):
    """
    旋转矩阵=>单位四元数
    """
    w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def batch_rotmat2quat_np(R):
    """

    :param R: (N,3,3)
    :return:
    """
    w = np.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return np.stack([w, x, y, z], axis=1)


def quat_mult(Q, P):
    """
    四元数乘法: 对于标准的两个四元数Q(4,)=(q0,q)、P(4,)=(p0,p)
    QP=(p0q0-p·q, p0q+q0p+q×p) [ ×为叉乘 ]

    :param Q: (N,4)
    :param P: (N,4)
    :return:
    """
    mul_w = Q[:, :1] * P[:, :1] - np.sum(Q[:, 1:] * P[:, 1:], axis=1, keepdims=True)
    mul_xyz = Q[:, :1] * P[:, 1:] + P[:, :1] * Q[:, 1:] + np.cross(Q[:, 1:], P[:, 1:], axis=1)
    return np.hstack([mul_w, mul_xyz])
#
#
# def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
#     """
#     depth: (H, W)
#     """
#     x = depth
#     x = np.nan_to_num(x)  # change nan to 0
#     mi = np.min(x)  # get minimum depth
#     ma = np.max(x)
#     x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
#     x = (255 * x).astype(np.uint8)
#     heat_img = cv2.applyColorMap(x, colormap=cmap)
#     return heat_img


def batch_quat2rotmat_np(q):
    """
    批量quat2rotmat
    :param q: (n,4)
    :return:
    """
    w, x, y, z = q[:, :1], q[:, 1:2], q[:, 2:3], q[:, 3:]
    R = np.hstack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
                   2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w,
                   2 * x * z - 2 * w * y, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2])
    return R.reshape((-1, 3, 3))


def quat_angle_diff(q1, q2):
    """
    这里给出了两种计算四元数角度差的方法
    1. 通过四元数乘法得到两个四元数的相对四元数，然后计算相对四元数的角度
    2. 通过四元数转旋转矩阵，然后计算旋转矩阵的trace，最后计算角度
    :param q1: (n,4)
    :param q2: (n,4)
    :return:
    """
    # 1.
    # inv_q1 = np.hstack([q1[:, :1], -q1[:, 1:]])
    # qd = quat_mult(inv_q1, q2)
    # angle = 2 * np.arctan2(np.linalg.norm(qd[:, 1:], axis=1), qd[:, 0]) / np.pi * 180
    # angle = np.minimum(angle, 360 - angle)

    # 2.
    R1, R2 = batch_quat2rotmat_np(q1), batch_quat2rotmat_np(q2)
    res_dot = np.matmul(R1, np.transpose(R2, axes=[0, 2, 1]))
    trace = np.trace(res_dot, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1) / 2,-1,1)) / np.pi * 180
    return angle
