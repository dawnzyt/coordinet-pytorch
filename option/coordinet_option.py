import argparse
import pickle
import os


class CoordiNetOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic
        self.parser.add_argument('--root_dir', type=str, default="./runs/coordinet", help='root dir of exp')
        self.parser.add_argument('--exp_name', type=str, default='church')
        self.parser.add_argument('--Description', type=str,
                                 default='1. 非固定efficientnet 2. Homosc loss 3. 启用ColorJitter 4. train数据集')

        # train
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--epochs', type=int, default=250)
        self.parser.add_argument('--last_epoch', type=int, default=0, help='>0 means continuing training')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--save_freq', type=int, default=93, help='save training model freq')
        self.parser.add_argument('--log_freq', type=int, default=1, help='log loss freq')
        self.parser.add_argument('--num_gpus', type=int, default=1)  # maybe pytorch_lighting use
        self.parser.add_argument('--ckpt_path', type=str,
                default='/root/coordinet/runs/coordinet/church/ckpt_epoch364_steps93_angle5.8764_t1.5458.pkl',
                help='ckpt path')

        # dataset
        self.parser.add_argument('--data_root_dir', type=str, default='/root/autodl-tmp/dataset/Cambridge',
                                 help='root dir of dataset')
        self.parser.add_argument('--scene', type=str, default='StMarysChurch', help='scene name')
        self.parser.add_argument('--reshape_size', type=int, default=320, help='reshape size')
        self.parser.add_argument('--crop_size', type=int, default=300, help='crop size')

        # coordinet model
        self.parser.add_argument('--backbone', type=str, default='efficientnet', choices=['efficientnet', 'resnet'],
                                 help='pretrained model')
        self.parser.add_argument('--backbone_path', type=str, default='/root/autodl-tmp/efficientnet-b3-5fb5a3c3.pth', )
        self.parser.add_argument('--fixed_weight', type=bool, default=False, help='if fixed weight of pretrained model')
        self.parser.add_argument('--feature_dim', type=int, default=1536,
                                 help='output feature dim of pretrained model like efficientnet-b3')
        self.parser.add_argument('--W', type=int, default=256, help='hidden dim of coordinet')
        self.parser.add_argument('--var_min', nargs='+', type=float, default=[0.01, 0.01, 0.01, 0.01],
                                 help='min var of t、R, totally depend on dataset bounding box')

        self.parser.add_argument('--learn_beta', type=bool, default=True, help='if learn beta in homosc loss')
        self.parser.add_argument('--loss_type', type=str, default='homosc', choices=['homosc', 'heterosc'],
                                 help='loss type of coordinet')

        ##############################
        ######   callbacks     #######
        self.parser.add_argument('--lr_decay_step', type=int, default=100, help='lr decay step')
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='lr decay rate')

    def into_opt(self, save_opt=True):
        opt = self.parser.parse_args()
        args = vars(opt)  # convert to dict
        exp_dir = os.path.join(opt.root_dir, opt.exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        if save_opt:
            # save opt txt
            with open(os.path.join(exp_dir, 'opt.txt'), 'w', encoding='utf-8') as f:
                f.write('--------------------Options-------------------\n')
                for k, v in sorted(args.items()):
                    f.write("%s: %s\n" % (str(k), str(v)))
                f.write('--------------------End-----------------------')
            # save opt pkl
            with open(os.path.join(exp_dir, 'opt.pkl'), 'wb') as f:
                pickle.dump(opt, f)
        return opt
