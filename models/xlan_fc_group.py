import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
import blocks

class XLAN_fc_group(AttBasicModel):
    def __init__(self):
        super(XLAN_fc_group, self).__init__()
        self.num_layers = 2
        # cfg.MODEL.BILINEAR.DECODE_LAYERS=2
        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        self.att_lstm = nn.LSTMCell(rnn_input_size+1024+1024, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        self.attention = blocks.create(            
            cfg.MODEL.BILINEAR.DECODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )

        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE),
            nn.GLU())

        self.att_lstm_fc = nn.LSTMCell(1024+1024+1024+1024+1024, cfg.MODEL.RNN_SIZE)

    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        fc_feats_group = kwargs[cfg.PARAM.FC_FEATS_GROUP]
        keywords = kwargs[cfg.PARAM.KEYWORDS]
        events=kwargs[cfg.PARAM.EVENTS]
        img_feat_mca = kwargs[cfg.PARAM.IMG_FEAT_MCA]
        lang_feat_mca = kwargs[cfg.PARAM.LANG_FEAT_MCA]
        ##
        gv_feat = img_feat_mca
        if gv_feat.shape[-1] == 1:  # empty gv_feat
             if att_mask is not None:
                 gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
             else:
                 gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed2(wt)
        feat_events=self.event_embed(events)
        feat_events_mean=torch.mean(feat_events, dim=1)
        # for fusing features
        prev_h = state[0][-1]
        h_att_group, c_att_group = self.att_lstm_fc(torch.cat([prev_h, fc_feats_group, xt, lang_feat_mca, feat_events_mean], 1))

        h_att, c_att = self.att_lstm(torch.cat([xt, lang_feat_mca, feat_events_mean,  gv_feat + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        ##
        ctx_input=torch.cat((att, h_att, h_att_group), dim=1)
        ##
        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))]

        return output, state