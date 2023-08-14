import glob
import os.path
import pickle
from PIL import Image
import numpy as np
import sys
from torch.utils.data import Dataset
import torch
import tqdm
from torchvision import transforms

from utils.utils import *

'''
https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
Each sequence (seq-XX.zip) consists of 500-1000 frames. Each frame consists of three files:
Color: frame-XXXXXX.color.png (RGB, 24-bit, PNG)
Depth: frame-XXXXXX.depth.png (depth in millimeters, 16-bit, PNG, invalid depth is set to 65535).
Pose: frame-XXXXXX.pose.txt (camera-to-world, 4×4 matrix in homogeneous coordinates).
Principle point (320,240), Focal length (585,585).
'''


class SevenScenesDataset(Dataset):
    def __init__(self, root_dir, scene='fire', split='train', reshape_size=320, crop_size=300):
        super().__init__()
        self.scene = scene
        self.root_dir = root_dir
        self.split = split
        self.reshape_size = reshape_size
        self.crop_size = crop_size
        self.train_tfms = transforms.Compose([transforms.Resize(320),
                                              transforms.RandomCrop(300),
                                              transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.test_tfms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.valid_tfms = transforms.Compose([transforms.Resize(320),
                                              transforms.CenterCrop(300),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.load_data()

    def load_data(self):
        print('load data of scene "{}", split: {}'.format(self.scene, self.split))
        base_dir = os.path.join(self.root_dir, self.scene)
        # train_split and test_split
        with open(os.path.join(base_dir, 'TrainSplit.txt'), 'r') as f:
            train_seq = f.readlines()
        with open(os.path.join(base_dir, 'TestSplit.txt'), 'r') as f:
            test_seq = f.readlines()
        train_seq = ['seq-0' + seq[-2] for seq in train_seq]  # seq-01 seq-02 ...
        test_seq = ['seq-0' + seq[-2] for seq in test_seq]  # ...

        def convert_to_rotation_matrix(R):
            # 使用奇异值分解对矩阵R进行分解
            U, _, Vt = np.linalg.svd(R)
            # 构造标准旋转矩阵
            R_std = np.dot(U, Vt)
            return R_std

        # 1. train split
        if self.split == 'train' or self.split == 'train_synthesis':
            self.train_paths = []
            self.train_poses = []
            for seq in train_seq:
                seq_dir = os.path.join(base_dir, seq)
                png_files = glob.glob(os.path.join(seq_dir, '*color.png'))
                dep_files = glob.glob(os.path.join(seq_dir, '*depth.png'))
                pose_files = glob.glob(os.path.join(seq_dir, '*.txt'))
                for png_file, dep_file, pose_file in zip(png_files, dep_files, pose_files):
                    self.train_paths.append(os.path.join(seq_dir, png_file))
                    c2w = np.loadtxt(os.path.join(seq_dir, pose_file), dtype=float)
                    w2c = np.linalg.inv(c2w)
                    R = w2c[:3, :3]
                    R = convert_to_rotation_matrix(w2c[:3, :3])
                    self.train_poses.append(torch.from_numpy(np.hstack([R, c2w[:3, -1:]])).float())
            self.train_poses = torch.stack(self.train_poses, dim=0)
        # 2. test split
        if self.split == 'valid' or self.split == 'test':
            self.test_paths = []
            self.test_poses = []
            for seq in test_seq:
                seq_dir = os.path.join(base_dir, seq)
                png_files = glob.glob(os.path.join(seq_dir, '*color.png'))
                dep_files = glob.glob(os.path.join(seq_dir, '*depth.png'))
                pose_files = glob.glob(os.path.join(seq_dir, '*.txt'))
                for png_file, dep_file, pose_file in zip(png_files, dep_files, pose_files):
                    self.test_paths.append(os.path.join(seq_dir, png_file))
                    c2w = np.loadtxt(os.path.join(seq_dir, pose_file), dtype=float)
                    w2c = np.linalg.inv(c2w)
                    R = w2c[:3, :3]
                    R = convert_to_rotation_matrix(w2c[:3, :3])
                    # print('det:', np.linalg.det(R), '\nmult:', R @ R.T)
                    self.test_poses.append(torch.from_numpy(np.hstack([R, c2w[:3, -1:]])).float())
            self.test_poses = torch.stack(self.test_poses, dim=0)
        # 3. synthesis split
        if self.split == 'synthesis' or self.split == 'train_synthesis':
            with open(os.path.join(base_dir, 'dataset_synthesis.txt'), 'r') as f:
                synthesis_split = f.readlines()  # synthesis split
            self.synthesis_paths = []
            self.synthesis_poses = []
            # dataset_synthesis.txt
            # Camera Position [X Y Z W P Q R] 同理
            for data in synthesis_split:
                data = data.split()
                local_path = data[0]
                self.synthesis_paths.append(os.path.join(base_dir, local_path))
                # pose
                params = np.array(data[1:], dtype=float)
                q = params[3:]
                q = q / np.linalg.norm(q)  # 归一化
                rotmat = quat2rotmat(q)  # rot: w2c
                center = params[:3].reshape(3, 1)  # c2w
                pose = torch.FloatTensor(np.hstack([rotmat, center]))
                self.synthesis_poses += [pose]
            self.synthesis_poses = torch.stack(self.synthesis_poses, dim=0)
        # 4. merge synthesis to train
        if self.split == 'train_synthesis':
            self.train_poses = torch.cat([self.train_poses, self.synthesis_poses], dim=0)
            self.train_paths = self.train_paths + self.synthesis_paths
        if self.split == 'synthesis':
            self.train_poses = self.synthesis_poses
            self.train_paths = self.synthesis_paths

    def __len__(self):
        if self.split == 'test' or self.split == 'valid':
            return len(self.test_paths)
        return len(self.train_paths)

    def __getitem__(self, idx):
        if self.split == 'test' or self.split == 'valid':
            path = self.test_paths[idx]
            return self.valid_tfms(Image.open(path)), self.test_poses[idx]
        path = self.train_paths[idx]
        return self.train_tfms(Image.open(path)), self.train_poses[idx]

    def uniform_crop(self, image_path, num_crops,pose):
        # 打开图像
        resize_tfms = transforms.Resize(self.reshape_size)
        image = resize_tfms(Image.open(image_path))
        width, height = image.size
        crop_width = self.crop_size
        crop_height = self.crop_size

        # 计算裁剪框之间的间隔
        stride_x = width // int(num_crops ** 0.5)
        stride_y = height // int(num_crops ** 0.5)

        crops = []
        # 使用均匀间隔遍历图像并裁剪
        for y in range(0, height - crop_height, stride_y):
            for x in range(0, width - crop_width, stride_x):
                crop = image.crop((x, y, x + crop_width, y + crop_height))
                crops.append(crop)
                # 当达到所需的裁剪数量时，跳出循环
                if len(crops) == num_crops:
                    break
            if len(crops) == num_crops:
                break
        # 将裁剪的图像转换为张量
        crops = [self.test_tfms(crop) for crop in crops]
        imgs = torch.stack(crops, dim=0)
        poses = torch.stack([pose] * len(crops), dim=0)
        return imgs, poses


if __name__ == '__main__':
    server_dir = '/root/autodl-tmp/dataset/7 scenes'
    scene = 'fire'
    local_dir = 'E:\\dataset\\7 scenes'
    dataset = SevenScenesDataset(server_dir, scene='fire', split='valid')
