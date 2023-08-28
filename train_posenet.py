import os
import sys

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset, DataLoader
from model.PoseNet import PoseNet, PoseLoss
from dataset.cambridge import CambridgeDataset
from option.posenet_option import PoseNetOption
from utils.utils import *
from utils.data_loger import DataLoger

if __name__ == '__main__':
    opt = PoseNetOption().into_opt()
    train_dataset = CambridgeDataset(opt.data_root_dir, opt.scene, split='train', img_size=opt.img_size, use_cache=True,
                                     if_save_cache=False)
    valid_dataset = CambridgeDataset(opt.data_root_dir, opt.scene, split='valid', img_size=opt.img_size, use_cache=True,
                                     if_save_cache=False)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True, num_workers=16)
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, shuffle=False, num_workers=16)
    loger=DataLoger(opt.root_dir,opt.exp_name)
    model = PoseNet().cuda()
    criterion = PoseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)
    epoch, steps, epoch_steps = 0, 0, 0


    for epoch in range(opt.epochs):
        for imgs, poses in train_dataloader:
            epoch_steps += 1
            steps += 1
            imgs, poses = imgs.cuda(), poses.cuda()
            t, q = model(imgs)
            y_q=torch.FloatTensor(batch_rotmat2quat_np(poses[:,:, :3].cpu().numpy())).cuda()
            loss,loss_t, loss_q = criterion(t, q, poses[:, :,-1]*s, y_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ##########################
            ######## diff
            q = q.detach().cpu().numpy()
            y_q = y_q.cpu().numpy()
            t = t.detach().cpu().numpy()
            y_t = poses[:, :,-1].cpu().numpy()*s

            angle_diff = np.mean(quat_angle_dif(q, y_q))
            t_diff = np.mean(np.linalg.norm(t - y_t, axis=1))
            norm_l1 = np.mean(np.linalg.norm(t - y_t, axis=1, ord=1))

            if steps % opt.log_freq == 0:
                print('[train] [epoch:%d ,step:%d] [loss]t:%.3f R:%.3f total:%.3f [diff]t:%.3fm,angle:%.3f°' % (
                    epoch, epoch_steps, loss_t, loss_q, loss.item(), t_diff, angle_diff))
                loger.log_loss({'train/loss_t': loss_t, 'train/loss_q': loss_q, 'train/loss': loss.item(),
                                'train/t_diff': t_diff, 'train/angle_diff': angle_diff}, epoch, steps, epoch_steps)
                # print('[Debug] [epoch:%d ,step:%d]' % (epoch, epoch_steps))
                # print('[T]', np.mean(t, axis=0))
                # print('[YT]', np.mean(y_t, axis=0))
            if steps % opt.save_freq == 0:
                ##########################
                ########## valid #########
                angle_diffs = []
                t_diffs = []
                for imgs,poses in valid_dataloader:
                    imgs, poses = imgs.cuda(), poses.cuda()
                    with torch.no_grad():
                        t, q = model(imgs,test_time=True)
                    q = q.detach().cpu().numpy()
                    y_q = batch_rotmat2quat_np(poses[:,:, :3].cpu().numpy())
                    t = t.detach().cpu().numpy()
                    y_t = poses[:, :,-1].cpu().numpy()*s
                    angle_diff = np.mean(quat_angle_diff(q, y_q))
                    t_diff = np.mean(np.linalg.norm(t - y_t, axis=1))
                    angle_diffs.append(angle_diff)
                    t_diffs.append(t_diff)
                angle_diff = np.mean(angle_diffs)
                t_diff = np.mean(t_diffs)
                print('[valid] [epoch:%d ,step:%d] [diff]t:%.3fm,angle:%.3f°' % (epoch, epoch_steps, t_diff, angle_diff))
                loger.log_loss({'valid/t_diff': t_diff, 'valid/angle_diff': angle_diff}, epoch, steps, epoch_steps)
                save_path = os.path.join(opt.root_dir, opt.exp_name, 'ckpt_epoch%d_steps%d_angle%.4f_t%.4f.pkl' % (
                    epoch, epoch_steps, angle_diff, t_diff))
                torch.save(model.state_dict(), save_path)
        scheduler.step()