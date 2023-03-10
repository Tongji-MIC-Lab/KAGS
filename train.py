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
from dataset import VISTDataset
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
from torch.utils.data import DataLoader
from torch.autograd import Variable
import opts
opt=opts.parse_opt()

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus>10
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda:1")

        self.rl_stage = False
        self.setup_logging()
        self.setup_network()
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

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

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
            model = models.create('XLAN_fc_group')

            if self.distributed:
                # this should be removed if we update BatchNorm stats
                self.model = torch.nn.parallel.DistributedDataParallel(
                    model.to(self.device),
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                    broadcast_buffers=False
                )
            else:
                self.model=model.cuda()

            if self.args.resume > 0:
                 self.model.load_state_dict(torch.load(self.snapshot_path("caption_model", self.args.resume),map_location=lambda storage, loc: storage))

            self.optim = Optimizer(self.model)
            self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
            self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, fc_feats, att_mask, keywords, events):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask,
            cfg.PARAM.FC_FEATS: fc_feats,
            cfg.PARAM.KEYWORDS: keywords,
            cfg.PARAM.EVENTS: events
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.ss_prob = ss_prob

    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    def forward(self, kwargs):
        if self.rl_stage == False:
            logit = self.model(**kwargs)
            max, max_pos=torch.max(logit, 2)
            gt=kwargs[cfg.PARAM.TARGET_SENT]
            loss, loss_info = self.xe_criterion(logit, gt)
        else:
            ids = kwargs[cfg.PARAM.TARGET_SENT]
            #ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
            fc_feats = kwargs[cfg.PARAM.FC_FEATS]

            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
            kwargs[cfg.PARAM.FC_FEATS]=fc_feats

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.decode(**kwargs)
            self.model.train()
            rewards_max, rewards_info_max = self.scorer(ids.data.cpu().numpy().tolist(), seq_max.data.cpu().numpy().tolist())

            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
            kwargs[cfg.PARAM.FC_FEATS]=fc_feats

            seq_sample, logP_sample = self.model.decode(**kwargs)
            rewards_sample, rewards_info_sample = self.scorer(ids.data.cpu().numpy().tolist(), seq_sample.data.cpu().numpy().tolist())

            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)
            
            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]

        return loss, loss_info

    def train(self):
        ## nn.parallel
        self.model.train()
        self.optim.zero_grad()
        dataset = VISTDataset(opt)
        dataset.set_option(data_type={'whole_story': False, 'split_story': True, 'caption': False})
        dataset.train()
        train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)


        iteration = 0
        print(cfg.SOLVER.MAX_EPOCH)
        for epoch in  range(cfg.SOLVER.MAX_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            #self.setup_loader(epoch)
            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            print('loader length : [%d]' %(len(train_loader)))
            for iter, batch in enumerate(train_loader):
                att_feats = Variable(batch['feature_obj']).cuda()
                fc_feats = Variable(batch['feature_fc']).cuda()
                keywords = Variable(batch['keywords']).cuda()
                events=Variable(batch['events']).cuda()
                input_seq = Variable(batch['split_story']).cuda()
                indices = batch['index']
                [B1, B2, obj_dim, fea_dim] = att_feats.size()
                [B1, B2, fea_dim] = fc_feats.size()
                [B1, B2, seqL] = input_seq.size()
                ######
                target_seq = input_seq
                input_seq0 = torch.zeros((B1, B2, 1), dtype=torch.long).cuda()
                target_seq = torch.cat((input_seq, input_seq0), dim=2)
                input_seq = torch.cat((input_seq0, input_seq), dim=2)
                ######
                att_mask = torch.ones(B1 * B2, obj_dim).cuda()
                gv_feat = torch.zeros(B1 * B2, 1).cuda()
                ####### resize
                input_seq = input_seq.view(B1 * B2, -1)
                target_seq = target_seq.view(B1 * B2, -1)
                att_feats = att_feats.view(B1 * B2, obj_dim, fea_dim)
                keywords=keywords.view(B1*B2, -1)
                events=events.view(B1*B2, -1)
                data_time.update(time.time() - start)

                kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, fc_feats, att_mask, keywords, events)
                loss, loss_info = self.forward(kwargs)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                self.optim.scheduler_step('Iter')
                
                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())
                ##
                if iteration % 20 == 0:
                    print('iteration: [%d], loss_avg: [%.4f]' %(iteration, losses.avg))
                    f1=open('./experiments/xlan/snapshot/log.txt', 'a+')
                    print('iteration: [%d], loss_avg: [%.4f]' %(iteration, losses.avg), file=f1)
                iteration += 1

                if self.distributed:
                    dist.barrier()
            self.save_model(epoch)
            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', type=str, default='./experiments/xlan')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume', type=int, default=-1)
    #if len(sys.argv) == 1:
    #parser.print_help()
    #sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()
