import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.util import TransformerWithToken_layoutganpp

from training.blip import init_tokenizer
from training.med import BertConfig, BertModel, BertLMHeadModel
from training.networks_stylegan2 import Encoder, Decoder

def merge_lists(lists):
    ret = []
    for l in lists:
        ret += l
    return ret

def split_list(list_a, chunk_size):
    ret = []
    for i in range(0, len(list_a), chunk_size):
        ret.append(list_a[i:i+chunk_size])
    return ret

def normalize_2nd_moment(x, eps=1e-8):
    return x * (x.square().mean(dim=1, keepdim=True) + eps).rsqrt()

class Generator(nn.Module):
    def __init__(self, z_dim, num_bbox_labels, img_channels, img_height, img_width, c_dim,
                 f_dim=256, num_heads=4, num_layers=8,
                 med_config='configs/med_config.json', bert_f_dim=768, bert_num_layers=12, bert_num_heads=12,
                 background_size=1024, im_f_dim=512):
        super().__init__()

        self.z_dim = z_dim
        self.num_bbox_labels = num_bbox_labels
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.fc_z = nn.Linear(z_dim*9, f_dim//2)
        #self.emb_label = nn.Embedding(num_bbox_labels, f_dim//2)

        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = bert_f_dim
        encoder_config.num_hidden_layers = bert_num_layers
        encoder_config.num_attention_heads = bert_num_heads
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        #self.im_encoder = Encoder(img_channels=img_channels, img_resolution=256, out_channels=im_f_dim, channel_max=im_f_dim, channel_base=8192, num_fp16_res=0, conv_clamp=None)
        self.bg_encoder = Encoder(img_channels=img_channels, img_resolution=background_size, out_channels=im_f_dim, channel_max=im_f_dim, channel_base=8192, num_fp16_res=0, conv_clamp=None)

        self.fc_in = nn.Linear(f_dim//2+768+1+im_f_dim, im_f_dim)

        te = nn.TransformerEncoderLayer(d_model=im_f_dim, nhead=num_heads,
                                        dim_feedforward=im_f_dim)
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(im_f_dim, 4)

    def forward(self, z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, c):
        B, N, C, H, W = bbox_patch.size()
        z = normalize_2nd_moment(z.view(B, -1)).unsqueeze(1).expand(-1, N, -1)
        z = self.fc_z(z)
        #l = self.emb_label(bbox_class)
        #aspect_ratio = (bbox_real[:,:,3] / bbox_real[:,:,2]).nan_to_num().unsqueeze(-1)
        text = self.tokenizer(merge_lists(bbox_text), padding='max_length', truncation=True, max_length=40, return_tensors="pt").to(bbox_class.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_feat = text_output.last_hidden_state[:,0,:].view(B, N, -1)
        text_len = torch.from_numpy(np.array([float(len(t))/40.0 for t in merge_lists(bbox_text)])).to(bbox_class.device).to(torch.float32).view(B, N, 1)
        #im_feat = self.im_encoder(bbox_patch.view(B*N, C, H, W)).view(B, N, -1)
        bg_feat = self.bg_encoder(background).unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([z, text_feat, text_len, bg_feat], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        x = torch.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_bbox_labels, img_channels, img_height, img_width, c_dim,
                 f_dim=256, num_heads=4, num_layers=8, max_bbox=50,
                 med_config='configs/med_config.json', bert_f_dim=768, bert_num_layers=12, bert_num_heads=12,
                 background_size=1024, im_f_dim=512):
        super().__init__()

        self.num_bbox_labels = num_bbox_labels
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # encoder
        self.fc_bbox = nn.Linear(4, f_dim//2)
        #self.emb_label = nn.Embedding(num_bbox_labels, f_dim//2)

        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = bert_f_dim
        encoder_config.num_hidden_layers = bert_num_layers
        encoder_config.num_attention_heads = bert_num_heads
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        #self.im_encoder = Encoder(img_channels=img_channels, img_resolution=256, out_channels=im_f_dim, channel_max=im_f_dim, channel_base=8192, num_fp16_res=0, conv_clamp=None)
        self.bg_encoder = Encoder(img_channels=img_channels, img_resolution=background_size, out_channels=im_f_dim, channel_max=im_f_dim, channel_base=8192, num_fp16_res=0, conv_clamp=None)

        self.enc_fc_in = nn.Linear(f_dim//2+768+1+im_f_dim, im_f_dim)

        self.enc_transformer = TransformerWithToken_layoutganpp(d_model=im_f_dim, nhead=num_heads,
                                                                dim_feedforward=im_f_dim,
                                                                num_layers=num_layers)

        self.fc_out_disc = nn.Linear(im_f_dim, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, f_dim))
        self.dec_fc_in = nn.Linear(f_dim+im_f_dim, im_f_dim)

        te = nn.TransformerEncoderLayer(d_model=im_f_dim, nhead=num_heads,
                                        dim_feedforward=im_f_dim)
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        #self.fc_out_cls = nn.Linear(im_f_dim, num_bbox_labels)
        self.fc_out_bbox = nn.Linear(im_f_dim, 4)

        # text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = im_f_dim
        decoder_config.num_hidden_layers = bert_num_layers
        decoder_config.num_attention_heads = bert_num_heads
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 

        # image decoder
        #self.im_decoder = Decoder(z_dim=im_f_dim, w_dim=im_f_dim, channel_max=im_f_dim, channel_base=8192, img_channels=img_channels, img_resolution=256, use_noise=False, num_fp16_res=0, conv_clamp=None, fused_modconv_default=False)
        self.bg_decoder = Decoder(z_dim=im_f_dim, w_dim=im_f_dim, channel_max=im_f_dim, channel_base=8192, img_channels=img_channels, img_resolution=background_size, use_noise=False, num_fp16_res=0, conv_clamp=None, fused_modconv_default=False)

    def forward(self, bbox, bbox_class, bbox_text, bbox_patch, padding_mask, background, c, reconst=False):
        B, N, C, H, W = bbox_patch.size()
        b = self.fc_bbox(bbox)
        #l = self.emb_label(bbox_class)
        text = self.tokenizer(merge_lists(bbox_text), padding='max_length', truncation=True, max_length=40, return_tensors="pt").to(bbox_class.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_feat = text_output.last_hidden_state[:,0,:].view(B, N, -1)
        text_len = torch.from_numpy(np.array([float(len(t))/40.0 for t in merge_lists(bbox_text)])).to(bbox_class.device).to(torch.float32).view(B, N, 1)
        #im_feat = self.im_encoder(bbox_patch.view(B*N, C, H, W)).view(B, N, -1)
        bg_feat = self.bg_encoder(background).unsqueeze(1).expand(-1, N, -1)
        x = self.enc_fc_in(torch.cat([b, text_feat, text_len, bg_feat], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)

        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        x0 = x[0]

        # logit_disc: [B,]
        logit_disc = self.fc_out_disc(x0).squeeze(-1)

        if not reconst:
            return logit_disc

        else:
            x = x0.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))

            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            x = x.permute(1, 0, 2)[~padding_mask]

            # logit_cls: [M, L]    bbox_pred: [M, 4]
            #logit_cls = self.fc_out_cls(x)
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

            # text decoder
            xx = x.unsqueeze(1) # [M, 1, C]
            decoder_input_ids = text.input_ids.clone() # [B*N, 40]
            decoder_input_ids = decoder_input_ids.view(B, N, -1)[~padding_mask] # [M, 40]
            decoder_input_ids[:,0] = self.tokenizer.bos_token_id # [M, 40]
            decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids==self.tokenizer.pad_token_id, -100) # [M, 40]
            decoder_output = self.text_decoder(decoder_input_ids, # [M, 40]
                                               attention_mask = text.attention_mask.view(B, N, -1)[~padding_mask], # [M, 40]
                                               encoder_hidden_states=xx, # [M, 1, C]
                                               encoder_attention_mask=torch.ones(xx.size()[:-1],dtype=torch.long).to(xx.device), # [M, 1]                 
                                               labels = decoder_targets, # [M, 40]
                                               return_dict = True,
                                               mode='text')
            loss_lm = decoder_output.loss

            # image and background decoder
            #im_rec = self.im_decoder(x)
            #im_rec[bbox_patch[~padding_mask]==0.0] = 0.0
            bg_rec = self.bg_decoder(x0)

            return logit_disc, bbox_pred, loss_lm, bg_rec
