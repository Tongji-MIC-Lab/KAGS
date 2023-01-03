# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import importlib
importlib.reload(sys)

import json
import h5py
import os
import os.path
import numpy as np
import random
import logging

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset


class VISTDataset(Dataset):
    def __init__(self, opt):
        self.mode = 'train'  # by default
        self.opt = opt

        self.task = opt.task  # option: 'story_telling', 'image_captioning'
        self.data_type = {
            'whole_story': False,
            'split_story': True,
            'caption': False
        }

        # open the hdf5 file
        print('DataLoader loading story h5 file: ', opt.story_h5)
        self.story_h5 = h5py.File(opt.story_h5, 'r', driver='core')['story']
        print("story's max sentence length is ", self.story_h5.shape[1])

        print('DataLoader loading story h5 file: ', opt.full_story_h5)
        self.full_story_h5 = h5py.File(opt.full_story_h5, 'r', driver='core')['story']
        print("full story's max sentence length is ", self.full_story_h5.shape[1])

        print('DataLoader loading description h5 file: ', opt.desc_h5)
        self.desc_h5 = h5py.File(opt.desc_h5, 'r', driver='core')['story']
        print("caption's max sentence length is ", self.desc_h5.shape[1])

        print('DataLoader loading story_line json file: ', opt.story_line_json)
        self.story_line = json.load(open(opt.story_line_json))

        self.id2word = self.story_line['id2words']
        print("vocab[0] = ", self.id2word['0'])
        print("vocab[1] = ", self.id2word['1'])
        self.word2id = self.story_line['words2id']
        self.vocab_size = len(self.id2word)
        print('vocab size is ', self.vocab_size)

        self.story_ids = {'train': [], 'val': [], 'test': []}
        self.description_ids = {'train': [], 'val': [], 'test': []}
        self.story_ids['train'] = self.story_line['train'].keys()
        self.story_ids['val'] = self.story_line['val'].keys()
        self.story_ids['test'] = self.story_line['test'].keys()
        self.description_ids['train'] = self.story_line['image2caption']['train'].keys()
        self.description_ids['val'] = self.story_line['image2caption']['val'].keys()
        self.description_ids['test'] = self.story_line['image2caption']['test'].keys()

        print('There are {} training data, {} validation data, and {} test data'.format(len(self.story_ids['train']),
                                                                                        len(self.story_ids['val']),
                                                                                        len(self.story_ids['test'])))

        ref_dir = 'data/reference'
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)

        # write reference files for storytelling
        for split in ['val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                if story['album_id'] not in reference:
                    reference[story['album_id']] = [story['origin_text']]
                else:
                    reference[story['album_id']].append(story['origin_text'])
            with open(os.path.join(ref_dir, '{}_reference.json'.format(split)), 'w') as f:
                json.dump(reference, f)

        # write reference files for captioning
        for split in ['val', 'test']:
            reference = {}
            #for flickr_id, story in self.story_line['image2caption_original'][split].iteritems():
            for flickr_id, story in self.story_line['image2caption_original'][split].items():
                reference[flickr_id] = story
            with open(os.path.join(ref_dir, '{}_desc_reference.json'.format(split)), 'w') as f:
                json.dump(reference, f)
        ####
        self.keywords=np.load('./story_files/keyfiles/story_keywords_vist_train_encode.npy', allow_pickle=True)[()]
        self.events=np.load('./story_files/keyfiles/event2id_vist.npy', allow_pickle=True)[()]
        img_id_all=[]
        for img_id in self.events:
            img_id_all.append(img_id)
        self.img_id_all=img_id_all

    def __getitem__(self, index):
        if self.task == 'story_telling':
            self.story_ids[self.mode]=list(self.story_ids[self.mode])
            story_id = self.story_ids[self.mode][index]
            story = self.story_line[self.mode][story_id]
            keywords=self.keywords[story_id]

            # load bu and fc feature
            feature_obj = np.zeros((story['length'], 36, self.opt.feat_size), dtype='float32')
            feature_fc = np.zeros((story['length'], self.opt.feat_size), dtype='float32')
            events = np.zeros((story['length'], 6), dtype=np.int)
            #feature_conv = np.zeros((story['length'], self.opt.conv_feat_size), dtype='float32')
            obj_root = './story_files/BU'
            fc_root = './story_files/fc'
            for i in range(story['length']):
                # load bu feature
                fc_path = os.path.join(obj_root, self.mode, '{}.npz'.format(story['flickr_id'][i]))
                obj_feature=np.load(fc_path)['feat']
                obj_feature=np.array(obj_feature).astype('float32')
                feature_obj[i] = obj_feature
                # load fc feature
                fc_path = os.path.join(fc_root, self.mode, '{}.npy'.format(story['flickr_id'][i]))
                feature_fc[i] = np.load(fc_path)
                # load events id
                idx=story['flickr_id'][i]
                if idx not in self.img_id_all:
                    event=np.zeros((6), dtype=np.int)
                else:
                    event=self.events[idx]
                events[i]=event
                
            ########
            sample = {'feature_obj': feature_obj, 'feature_fc': feature_fc, 'keywords': keywords, 'events': events}

            # load story
            if self.data_type['whole_story']:
                whole_story = self.full_story_h5[story['whole_text_index']]
                sample['whole_story'] = np.int64(whole_story)

            if self.data_type['split_story']:
                split_story = self.story_h5[story['text_index']]
                sample['split_story'] = np.int64(split_story)

            # load caption
            if self.data_type['caption']:
                caption = []
                for flickr_id in story['flickr_id']:
                    if flickr_id in self.story_line['image2caption'][self.mode]:
                        descriptions = self.story_line['image2caption'][self.mode][flickr_id]['caption']
                        random_idx = np.random.choice(len(descriptions), 1)[0]
                        caption.append(self.desc_h5[descriptions[random_idx]])
                    else:
                        caption.append(np.zeros((self.desc_h5.shape[1],), dtype='int64'))
                sample['caption'] = np.asarray(caption, 'int64')
            sample['index'] = np.int64(index)

            return sample

        elif self.task == "image_captioning":
            flickr_id = self.description_ids[self.mode][index]
            descriptions = self.story_line['image2caption'][self.mode][flickr_id]['caption']
            random_idx = np.random.choice(len(descriptions), 1)[0]
            description = descriptions[random_idx]

            fc_path = os.path.join(self.opt.data_dir, 'resnet_features/fc', self.mode,
                                   '{}{}.npy'.format(self.opt.prefix, flickr_id))
            conv_path = os.path.join(self.opt.data_dir, 'resnet_features/conv', self.mode, '{}.npy'.format(flickr_id))
            feature_fc = np.load(fc_path)
            feature_conv = np.load(conv_path).flatten()

            sample = {'feature_fc': feature_fc, 'feature_conv': feature_conv}
            target = np.int64(self.desc_h5[description])
            sample['whole_story'] = target
            sample['mask'] = np.zeros_like(target, dtype='float32')
            nonzero_num = (target != 0).sum() + 1
            sample['mask'][:nonzero_num] = 1
            sample['index'] = np.int64(index)

            return sample

    def __len__(self):
        if self.task == 'story_telling':
            return len(self.story_ids[self.mode])
        elif self.task == 'image_captioning':
            return len(self.description_ids[self.mode])
        else:
            raise Exception("{} task is not proper for this dataset.".format(self.task))

    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def test(self):
        self.mode = 'test'

    def set_option(self, data_type=None):
        if self.task == 'story_telling':
            if data_type is not None:
                self.data_type = data_type
        else:
            pass
            # logging.error("{} task is not proper for this dataset.".format(self.task))
            # raise Exception("{} task is not proper for this dataset.".format(self.task))

    def get_GT(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            story = self.story_line[self.mode][story_id]
            return story['origin_text']
        elif self.task == 'image_captioning':
            raise Exception("To be implemented.")
        else:
            raise Exception("{} task is not proper for this dataset.".format(self.task))

    def get_id(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            return self.story_line[self.mode][story_id]['album_id'], self.story_line[self.mode][story_id]['flickr_id']
        else:
            return self.description_ids[self.mode][index]

    def get_all_id(self, index):
        story_id = self.story_ids[self.mode][index]
        return self.story_line[self.mode][story_id]['album_id'], self.story_line[self.mode][story_id]['flickr_id']

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.id2word

    def get_word2id(self):
        return self.word2id

    def get_whole_story_length(self):
        return self.full_story_h5.shape[1]

    def get_story_length(self):
        return self.story_h5.shape[1]

    def get_caption_length(self):
        return self.desc_h5.shape[1]


if __name__ == "__main__":
    import sys
    import os
    import opts
    import time

    start = time.time()

    opt = opts.parse_opt()
    dataset = VISTDataset(opt)

    print("dataset finished: ", time.time() - start)
    start = time.time()

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=opt.shuffle, num_workers=8)

    print("dataloader finished: ", time.time() - start)

    dataset.train()  # train() mode has to be called before using train_loader
    for iter, batch in enumerate(train_loader):
        print("enumerate: ", time.time() - start)
        print(iter)
        print(batch[0].size())
