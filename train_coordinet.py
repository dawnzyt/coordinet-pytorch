import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from model.coordinet import CoordiNetSystem
from utils.data_loger import DataLoger
from utils.utils import *

from dataset.cambridge import CambridgeDataset
from dataset.seven_scenes import SevenScenesDataset
from option.coordinet_option import CoordiNetOption

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    opt = CoordiNetOption().into_opt()
    # custom dataset
    train_dataset = SevenScenesDataset(opt.data_root_dir, opt.scene, 'train_synthesis', opt.reshape_size, opt.crop_size)
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=16, shuffle=True)
    valid_dataset = SevenScenesDataset(opt.data_root_dir, opt.scene, 'valid', opt.reshape_size, opt.crop_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers=16, shuffle=False)
    loger = DataLoger(root_dir=opt.root_dir, exp_name=opt.exp_name)
    # init model
    coordinet = CoordiNetSystem(opt.feature_dim, opt.W, opt.loss_type, opt.learn_beta, opt.var_min, opt.fixed_weight,
                                opt.backbone,opt.backbone_path)
    coordinet = coordinet.cuda()
    # init optimizer
    optimizer = Adam(coordinet.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)
    # resume
    steps, epoch_steps = 0, 0
    if opt.last_epoch:
        ckpt = torch.load(opt.ckpt_path)
        coordinet.load_state_dict(ckpt['coordinet'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch_steps = ckpt['epoch_steps']
        steps += len(dataloader) * (opt.last_epoch - 1) + epoch_steps
        print('resume from epoch:{}, epoch_steps:{}, ckpt_path:{}'.format(opt.last_epoch, epoch_steps, opt.ckpt_path))
    # train
    start_epoch = opt.last_epoch if (epoch_steps and epoch_steps % len(dataloader) != 0) else opt.last_epoch + 1
    print('len(dataloader):', len(dataloader))
    # s = float(valid_dataset.scale_factor)
    for epoch in range(start_epoch, opt.epochs + 1):
        if epoch != opt.last_epoch:
            epoch_steps = 0
        train_t_diffs = []
        train_angle_diffs = []
        for i, (imgs, poses) in enumerate(dataloader):
            epoch_steps += 1
            steps += 1
            imgs = imgs.cuda()
            poses = poses.cuda()
            poses[:, :, -1] = poses[:, :, -1]
            # forward
            t, q, v, losses = coordinet.forward(imgs, cal_loss=True, y_pose=poses)
            loss, loss_t, loss_R, t_diff, angle_diff = losses['loss'], losses['t'], losses['R'], losses['t_diff'], \
                losses['angle_diff']
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ########################################
            ############## log loss ################
            if steps % opt.log_freq == 0:
                t_diff, angle_diff = torch.mean(t_diff).item(), torch.mean(angle_diff).item() / np.pi * 180
                train_t_diffs.append(t_diff)
                train_angle_diffs.append(angle_diff)
                loger.log_loss({'train/loss_t': loss_t, 'train/loss_R': loss_R, 'train/loss': loss.item(),
                                'train/t_diff': t_diff, 'train/angle_diff': angle_diff}, epoch, steps, epoch_steps)
                print('[train] [epoch:%d, steps:%d] [loss]t:%.4f R:%.4f total:%.4f [diff] t:%.4fm angle:%.4f°' % (
                    epoch, epoch_steps, loss_t, loss_R, loss.item(), t_diff, angle_diff))
            ########################################
            ############## save ckpt ###############
            if steps % opt.save_freq == 0:
                ##########################
                ########## valid #########
                coordinet.eval()  # !!!!!!!!!!!!!!必须加无效化dropout!!!!!!!!!!!!!!!!!!!!
                t_diffs = []
                angle_diffs = []
                for imgs, poses in tqdm(valid_dataloader, total=len(valid_dataloader), file=sys.stdout):
                    imgs = imgs.cuda()
                    poses = poses.cuda()
                    poses[:, :, -1] = poses[:, :, -1]
                    with torch.no_grad():
                        t, q, _, losses = coordinet.forward(imgs, cal_loss=True, y_pose=poses)
                    t_diffs.append(losses['t_diff'].cpu().numpy())
                    angle_diffs.append(losses['angle_diff'].cpu().numpy())
                angle_diff = np.mean(np.hstack(angle_diffs)) / np.pi * 180
                t_diff = np.mean(np.hstack(t_diffs))
                print('[valid] [epoch:%d, mean_diff]t_diff:%.4fm, angle_diff:%.4f°' % (epoch, t_diff, angle_diff))
                print('[train] [epoch:%d, mean_diff]t_diff:%.4fm, angle_diff:%.4f°' % (
                    epoch, np.mean(train_t_diffs), np.mean(train_angle_diffs)))
                print('beta:', coordinet.loss_fn.beta)
                ##########################
                ########## save ##########
                loger.log_loss({'valid/angle_diff': angle_diff, 'valid/t_diff': t_diff}, epoch, steps, epoch_steps)
                ckpt = {'coordinet': coordinet.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch_steps': epoch_steps}
                save_path = os.path.join(opt.root_dir, opt.exp_name, 'ckpt_epoch%d_steps%d_angle%.4f_t%.4f.pkl' % (
                    epoch, epoch_steps, angle_diff, t_diff))
                torch.save(ckpt, save_path)
                coordinet.train()
            if epoch_steps == len(dataloader):
                break
        scheduler.step()
