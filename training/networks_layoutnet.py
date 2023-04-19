'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu

 * Modified from LayoutGAN++ repo: https://github.com/ktrk115/const_layout
'''

import torch
import torch.nn as nn

from training.util import TransformerWithToken_layoutganpp


class LayoutNet(nn.Module):
    def __init__(self, num_label):
        super().__init__()

        d_model = 256
        nhead = 4
        num_layers = 4
        max_bbox = 50

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken_layoutganpp(d_model=d_model,
                                                                dim_feedforward=d_model // 2,
                                                                nhead=nhead, num_layers=num_layers)

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, padding_mask, label_idx_replace=False, label_idx_replace_2=False):
        b = self.fc_bbox(bbox)
        if label_idx_replace:
            label[label<=4] = 2 # {'header', 'pre-header', 'post-header', 'body text', 'disclaimer / footnote'} -> 'TEXT'
            label[label==5] = 4 # 'button' -> 'BUTTON'
            label[label==7] = 3 # 'logo' -> 'PICTOGRAM'
            label[label==6] = 7 # 'callout' -> 'ADVERTISEMENT'
        elif label_idx_replace_2:
            label[label==0] = 3 # 'Logo' -> 'PICTOGRAM'
            label[label==3] = 3 # 'Embellishment' -> 'PICTOGRAM'
            label[label==2] = 444 # 'Underlay' -> 'BUTTON'
            label[label==4] = 2 # 'Highlighted text' -> 'TEXT'
            label[label==444] = 4 # 'Underlay' -> 'BUTTON'
            label[label==1] = 2 # 'Text' -> 'TEXT'
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]

    def forward(self, bbox, label, padding_mask):
        B, N, _ = bbox.size()
        x = self.extract_features(bbox, label, padding_mask)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)[~padding_mask]

        # logit_cls: [M, L]    bbox_pred: [M, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        return logit_disc, logit_cls, bbox_pred
