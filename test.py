import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler_round import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        self.evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        ##### for fc conv module
        model = models.create('XLAN_fc_group')
        self.model=model.cuda()
           
                   
    def eval(self, epoch):
        epoch0=20
        for i in range(30):
            print('Now is in the round: [%d]' %(i+epoch0))
            epoch=epoch0 + i*1
            
            ## for multiple gpus
#            pretrained_dict=torch.load(self.snapshot_path("caption_model", epoch))
#            net_dict=self.model.state_dict()
#            for key, value in pretrained_dict.items():
#                net_dict[key[7:]]=value
#                net_dict.update(net_dict)
#                self.model.load_state_dict(net_dict)
            ## for single gpu
            self.model.load_state_dict(torch.load(self.snapshot_path("caption_model", epoch),map_location=lambda storage, loc: storage))
            res = self.evaler(self.model, 'test_' + str(epoch))

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument("--folder", dest='folder', default='./experiments/xlan', type=str)
    parser.add_argument("--resume", type=int, default=-1)
    ########################
    ########################
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(args.resume)
