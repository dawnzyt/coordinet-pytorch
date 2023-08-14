import argparse
import pickle
import os


class PoseNetOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic
        self.parser.add_argument('--root_dir', type=str, default="./runs/posenet", help='root dir of exp')
        self.parser.add_argument('--exp_name', type=str, default='StMarysChurch2')
        self.parser.add_argument('--Description', type=str,
                                 default='posenet测试')

        # train
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--epochs', type=int, default=400)
        self.parser.add_argument('--last_epoch', type=int, default=0, help='>0 means continuing training')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--save_freq', type=int, default=154, help='save training model freq')
        self.parser.add_argument('--log_freq', type=int, default=1, help='log loss freq')
        self.parser.add_argument('--num_gpus', type=int, default=1)  # maybe pytorch_lighting use
        self.parser.add_argument('--ckpt_path', type=str,
                                 default='',
                                 help='ckpt path')

        # dataset
        self.parser.add_argument('--data_root_dir', type=str, default='/root/autodl-tmp/dataset/Cambridge',
                                 help='root dir of dataset')
        self.parser.add_argument('--scene', type=str, default='StMarysChurch', help='scene name')
        self.parser.add_argument('--img_size', nargs='+', type=int, default=(640, 360), help='image size')
        self.parser.add_argument('--use_cache', type=bool, default=True, help='if use cache when loading dataset')

        # coordinet model
        self.parser.add_argument('--backbone_path', type=str, default='/root/autodl-tmp/efficientnet-b3-5fb5a3c3.pth', )
        self.parser.add_argument('--feature_dim', type=int, default=1536,
                                 help='feature dim of pretrained model like efficientnet-b3')
        self.parser.add_argument('--W', type=int, default=256, help='hidden dim of coordinet')
        self.parser.add_argument('--var_min', nargs='+', type=float, default=[0.001987, 1.38e-5, 0.003532, 0.004874],
                                 help='min var of t、R, totally depend on dataset bounding box')
        # self.parser.add_argument('--var_min', type=float, default=0.0314, help='min var of coordinet R vector')

        ##############################
        ######   callbacks     #######
        self.parser.add_argument('--lr_decay_step', type=int, default=50, help='lr decay step')
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
