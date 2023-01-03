import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import misc.utils as utils
#import datasets.data_loader as data_loader
from lib.config import cfg
from dataset2_fc_conv_comsense import VISTDataset
import opts
opt=opts.parse_opt()
from torch.utils.data import DataLoader
from torch.autograd import Variable
from vist_eval.album_eval import AlbumEvaluator
dataset2 = VISTDataset(opt)

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        eval_annfile
    ):
        super(Evaler, self).__init__()
        self.vocab = dataset2.get_vocab()
        self.test_loader = DataLoader(dataset2, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask, fc_feats, keywords):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.FC_FEATS]=fc_feats
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        kwargs[cfg.PARAM.KEYWORDS]=keywords
        return kwargs

        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        predictions = {}
        ##
        txt_root='./prediction_results/Xlan_fc_conv_comsense_v1_round_12322/predict_'
        txt_root2=txt_root + rname
        prediction_txt = open(txt_root2, 'w')  # open the file to store the predictions
        with torch.no_grad():
            len_test=len(self.test_loader)
            id=0
            for iter, batch in enumerate(self.test_loader):
                att_feats = Variable(batch['feature_obj']).cuda()
                fc_feats=Variable(batch['feature_fc']).cuda()
                input_seq = Variable(batch['split_story']).cuda()
                indices = batch['index'].numpy()
                keywords=Variable(batch['keywords']).cuda()
                #########
                [B1, B2, obj_dim, fea_dim] = att_feats.size()
                [B1, B2, seqL]=input_seq.size()
                ##########
                att_mask=torch.ones(B1*B2, obj_dim).cuda()
                gv_feat=torch.zeros(B1*B2, 1).cuda()
                att_feats=att_feats.view(B1*B2, obj_dim, fea_dim)
                keywords=keywords.view(B1*B2, -1).cuda()
                ids=indices
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask, fc_feats, keywords)
                if kwargs['BEAM_SIZE'] > 1:
                    #seq, _ = model.module.decode_beam(**kwargs)
                    seq, _ = model.decode_beam(**kwargs)
                else:
                    #seq, _ = model.module.decode(**kwargs)
                    print('beam search is 1')
                    seq, _ = model.decode(**kwargs)
                
                [B, seqL]=seq.size()
                results=seq.view(-1, 5, seqL)
                ##########
                stories = utils.decode_story(dataset2.get_vocab(), results)
                story_len=len(stories)
                for j, story in enumerate(stories):
                    vid, _ = dataset2.get_id(indices[j])
                    if vid not in predictions:  # only predict one story for an album
                        # write into txt file for evaluate metrics like Cider
                        id=id+1
                        prediction_txt.write('{}\t {}\n'.format(vid, story))
                        # save into predictions
                        predictions[vid] = story

            prediction_txt.close()
        return len_test
