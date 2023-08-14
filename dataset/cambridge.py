import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.utils import *


# split∈{"train","train_synthesis","synthesis","test","valid"}
class CambridgeDataset(Dataset):
    def __init__(self, root_dir, scene, split='train', reshape_size=320, crop_size=300):
        super().__init__()
        self.scene = scene
        self.root_dir = root_dir
        self.split = split
        self.reshape_size = reshape_size
        self.crop_size = crop_size
        self.train_tfms = transforms.Compose([transforms.Resize(reshape_size),
                                              transforms.RandomCrop(crop_size),
                                              transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.test_tfms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.valid_tfms = transforms.Compose([transforms.Resize(reshape_size),
                                              transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.load_data()

    def load_data(self):
        print('load image and pose data of scene "{}", split: {}'.format(self.scene, self.split))
        base_dir = os.path.join(self.root_dir, self.scene)
        self.scale_factor = np.loadtxt(os.path.join(base_dir, 'scale_factor.txt'))  # load scale factor
        with open(os.path.join(base_dir, 'dataset_train.txt'), 'r') as f:
            train_split = f.readlines()[3:]  # train split
        with open(os.path.join(base_dir, 'dataset_test.txt'), 'r') as f:
            test_split = f.readlines()[3:]  # test split
        with open(os.path.join(base_dir, 'dataset_synthesis.txt'), 'r') as f:
            synthesis_split = f.readlines()  # synthesis split
        # 1. load train
        # dataset_train.txt
        # Camera Position [X Y Z W P Q R], 其中[W,P,Q,R]是w2c,[X,Y,Z]是c2w即camara position
        self.train_paths = []
        if self.split == 'train' or self.split == 'train_synthesis':
            self.train_w2cs = []
            self.train_imgs = []
            for data in tqdm.tqdm(train_split, total=len(train_split), file=sys.stdout, desc='load train data'):
                data = data.split()
                local_path = data[0]
                # pose
                params = np.array(data[1:], dtype=float)
                q = params[3:]
                q = q / np.linalg.norm(q)  # 归一化
                params[3:] = q
                rotmat = quat2rotmat(q)  # rot: w2c
                center = params[:3].reshape(3, 1)  # c2w
                t_vec = center  # c2w
                # t_vec = -rotmat @ center  # w2c
                w2c = torch.FloatTensor(np.hstack([rotmat, t_vec]))
                c2w = torch.FloatTensor(np.hstack([np.linalg.inv(rotmat), t_vec]))
                self.train_w2cs += [w2c]
                # train_w2cs += [torch.FloatTensor(params)]
                self.train_paths.append(os.path.join(base_dir, local_path))
            self.train_poses = torch.stack(self.train_w2cs, dim=0)
        # 2. load test
        # dataset_test.txt
        # Camera Position [X Y Z W P Q R], 其中[W,P,Q,R]是w2c,[X,Y,Z]是c2w即camara position
        self.test_paths = []
        if self.split == 'test' or self.split == 'valid':
            self.test_w2cs = []
            for data in tqdm.tqdm(test_split, total=len(test_split), file=sys.stdout, desc='load test data'):
                data = data.split()
                local_path = data[0]
                # pose
                params = np.array(data[1:], dtype=float)
                q = params[3:]
                q = q / np.linalg.norm(q)  # 归一化
                params[3:] = q
                rotmat = quat2rotmat(q)  # rot: w2c
                center = params[:3].reshape(3, 1)
                t_vec = center  # c2w
                # t_vec = -rotmat @ center # w2c
                w2c = torch.FloatTensor(np.hstack([rotmat, t_vec]))
                c2w = torch.FloatTensor(np.hstack([np.linalg.inv(rotmat), t_vec]))
                self.test_w2cs += [w2c]
                # test_w2cs += [torch.FloatTensor(params)]
                self.test_paths.append(os.path.join(base_dir, local_path))
            self.test_poses = torch.stack(self.test_w2cs, dim=0)
            # print(self.test_imgs.shape)
        # 3. load synthesis
        # dataset_synthesis.txt
        # Camera Position [X Y Z W P Q R] 同理
        self.synthesis_paths = []
        if self.split == 'synthesis' or self.split == 'train_synthesis':
            self.synthesis_w2cs = []
            for data in tqdm.tqdm(synthesis_split, total=len(synthesis_split), file=sys.stdout,
                                  desc='load synthesis data'):
                data = data.split()
                local_path = data[0]
                # pose
                params = np.array(data[1:], dtype=float)
                q = params[3:]
                q = q / np.linalg.norm(q)  # 归一化
                params[3:] = q
                rotmat = quat2rotmat(q)  # rot: w2c
                center = params[:3].reshape(3, 1)  # c2w
                t_vec = center  # c2w
                # t_vec = -rotmat @ center  # w2c
                w2c = torch.FloatTensor(np.hstack([rotmat, t_vec]))
                c2w = torch.FloatTensor(np.hstack([np.linalg.inv(rotmat), t_vec]))
                self.synthesis_w2cs += [w2c]
                # synthesis_w2cs += [torch.FloatTensor(params)]
                self.synthesis_paths.append(os.path.join(base_dir, local_path))
            self.synthesis_poses = torch.stack(self.synthesis_w2cs, dim=0)
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
        if self.split == 'valid':
            path = self.test_paths[idx]
            return self.valid_tfms(Image.open(path)), self.test_poses[idx]
        path = self.train_paths[idx]
        return self.train_tfms(Image.open(path)), self.train_poses[idx]

    def uniform_crop(self, image_path, num_crops, pose):
        # 打开图像
        resize_tfms = transforms.Resize(self.reshape_size)
        image = resize_tfms(Image.open(image_path))
        width, height = image.size
        crop_width = self.crop_size
        crop_height = self.crop_size

        # 计算裁剪框之间的间隔
        stride_x = (width - crop_width) // int(num_crops ** 0.5)
        stride_y = (height - crop_height) // int(num_crops ** 0.5)
        # print('size(%d %d)->size(%d %d), stride=(%d %d)' % (width, height, crop_width, crop_height, stride_x, stride_y))
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
    data_root_dir = '/root/autodl-tmp/dataset/Cambridge'
    scene = 'StMarysChurch'
    # scene='KingsCollege'
    # dataset = CambridgeDataset(data_root_dir, scene, 'synthesis', (640, 360), use_cache=False, if_save_cache=True)
    dataset = CambridgeDataset(data_root_dir, scene, 'train_synthesis')
    _ = CambridgeDataset(data_root_dir, scene, 'valid')
