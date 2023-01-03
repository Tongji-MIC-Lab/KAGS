import os
import sys
import numpy as np
import pickle
from lib.config import cfg

from scorer.cider import Cider
from scorer.bleu import Bleu

#factory = {
#    'Bleu': Bleu,
#    'CIDEr': Cider
#}
factory = {
    'CIDEr': Cider
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words

class Scorer(object):
    def __init__(self):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        #cfg.SCORER.TYPES=['Bleu']
        for name in cfg.SCORER.TYPES:
            print('we are now optimizing %s' % (name))
            self.scorers.append(factory[name]())

    def __call__(self, gt, res):
        hypo = [get_sents(r) for r in res]
        gts = [get_sents(j) for j in gt]
        #gts = [self.gts[i] for i in ids]

        rewards_info = {}
        #[B,C]=gts.size()
        rewards = np.zeros(len(gts))    #### 50 means the batch size
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo)
            rewards += self.weights[i] * scores
            #rewards +=  np.mean(scores,0)
            rewards_info[cfg.SCORER.TYPES[i]] = score
        return rewards, rewards_info