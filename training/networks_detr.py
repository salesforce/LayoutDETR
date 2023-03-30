import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.util import TransformerWithToken_layoutganpp

from training.blip import init_tokenizer
from training.med import BertConfig, BertModel, BertLMHeadModel
from training.networks_stylegan2 import Decoder

from detr_util.misc import nested_tensor_from_tensor_list
from training.detr_position_encoding import PositionEmbeddingSine
from training.detr_backbone import Backbone, Joiner
from training.detr_transformer import Transformer, TransformerWithToken, TransformerDecoderLayer, TransformerDecoder

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

def build_backbone():
    backbone = Backbone(name='resnet50', train_backbone=True, return_interm_layers=None, dilation=False)
    position_embedding = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, num_bbox_labels, img_channels, img_height, img_width, c_dim,
                 f_dim=256, num_heads=4, num_layers=8,
                 hidden_dim=256,
                 med_config='configs/med_config.json', bert_f_dim=768, bert_num_encoder_layers=12, bert_num_decoder_layers=12, bert_num_heads=12,
                 background_size=1024, im_f_dim=512,
                 max_text_length=256):
        super().__init__()
        self.z_dim = z_dim
        self.num_bbox_labels = num_bbox_labels
        self.c_dim = c_dim
        self.max_text_length = max_text_length

        ####################################
        # generator
        ####################################
        self.backbone = build_backbone()
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        self.fc_z = nn.Linear(z_dim*9, bert_f_dim)
        self.emb_label = nn.Embedding(num_bbox_labels, bert_f_dim)

        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = bert_f_dim
        encoder_config.num_hidden_layers = bert_num_encoder_layers
        encoder_config.num_attention_heads = bert_num_heads
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        self.enc_text_len = nn.Embedding(max_text_length, bert_f_dim)

        self.fc_in = MLP(input_dim=bert_f_dim+bert_f_dim+bert_f_dim+bert_f_dim, hidden_dim=bert_f_dim, output_dim=hidden_dim, num_layers=3)

        self.transformer = Transformer(
                                d_model=hidden_dim,
                                dropout=0.1,
                                nhead=8,
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                normalize_before=False,
                                return_intermediate_dec=False,
                            )
        #self.bbox_embed = nn.Linear(hidden_dim, 4)
        self.bbox_embed = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=4, num_layers=3)

        ####################################
        # reconstructor
        ####################################
        # noise decoder
        #self.z_rec_transformer = TransformerWithToken_layoutganpp(d_model=hidden_dim, dim_feedforward=2048, nhead=8, num_layers=1)
        self.fc_z_rec = nn.Linear(hidden_dim, z_dim*9)

        # label decoder
        self.fc_out_cls = nn.Linear(hidden_dim, num_bbox_labels)

        # text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = im_f_dim
        decoder_config.num_hidden_layers = bert_num_decoder_layers
        decoder_config.num_attention_heads = bert_num_heads
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # text length decoder
        self.fc_text_len_rec = nn.Linear(hidden_dim, max_text_length)

    def forward(self, z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, c, reconst=False):
        if isinstance(background, (list, torch.Tensor)):
            background = nested_tensor_from_tensor_list(background)
        bg_feat, pos = self.backbone(background)
        bg_feat, mask = bg_feat[-1].decompose()
        assert mask is not None

        B, N, C, H, W = bbox_patch.size()
        z0 = normalize_2nd_moment(z.view(B, -1))
        z = self.fc_z(z0).unsqueeze(1).expand(-1, N, -1)
        l = self.emb_label(bbox_class)
        #aspect_ratio = (bbox_real[:,:,3] / bbox_real[:,:,2]).nan_to_num().unsqueeze(-1)
        text = self.tokenizer(merge_lists(bbox_text), padding='max_length', truncation=True, max_length=self.max_text_length, return_tensors="pt").to(bbox_class.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_feat = text_output.last_hidden_state[:,0,:].view(B, N, -1)
        #text_len = torch.from_numpy(np.array([float(len(t))/40.0 for t in merge_lists(bbox_text)])).to(bbox_class.device).to(torch.float32).view(B, N, 1)
        text_len = torch.from_numpy(np.array([len(t) for t in merge_lists(bbox_text)])).to(bbox_class.device).to(torch.int64).view(B, N)
        text_len_feat = self.enc_text_len(text_len)
        x = torch.cat([z, l, text_feat, text_len_feat], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)
        
        x = self.transformer(src=self.input_proj(bg_feat), mask=mask, pos_embed=pos[-1], tgt=x, tgt_key_padding_mask=padding_mask)[0]
        bbox_fake = self.bbox_embed(x).sigmoid()

        if not reconst:
            return bbox_fake

        else:
            # noise reconstruction
            z_rec = self.fc_z_rec(x[~padding_mask])
            loss_z = F.mse_loss(z_rec, z0.unsqueeze(1).expand(-1, N, -1)[~padding_mask])

            # label reconstruction
            logit_cls = self.fc_out_cls(x[~padding_mask])

            # text reconstruction
            xx = x[~padding_mask].unsqueeze(1) # [M, 1, C]
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

            # text length reconstruction
            text_len_rec = self.fc_text_len_rec(x[~padding_mask])
            loss_text_len = F.cross_entropy(text_len_rec, text_len[~padding_mask])

            return bbox_fake, loss_z, logit_cls, loss_lm, loss_text_len


class Discriminator(nn.Module):
    def __init__(self, num_bbox_labels, img_channels, img_height, img_width, c_dim,
                 f_dim=256, num_heads=4, num_layers=8, max_bbox=50,
                 hidden_dim=256,
                 med_config='configs/med_config.json', bert_f_dim=768, bert_num_encoder_layers=12, bert_num_decoder_layers=12, bert_num_heads=12,
                 background_size=1024, im_f_dim=512,
                 max_text_length=256):
        super().__init__()

        self.num_bbox_labels = num_bbox_labels
        self.c_dim = c_dim
        self.max_text_length = max_text_length

        ####################################
        # encoder
        ####################################
        self.backbone = build_backbone()
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        self.fc_bbox = nn.Linear(4, bert_f_dim)
        self.emb_label = nn.Embedding(num_bbox_labels, bert_f_dim)

        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = bert_f_dim
        encoder_config.num_hidden_layers = bert_num_encoder_layers
        encoder_config.num_attention_heads = bert_num_heads
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        self.enc_text_len = nn.Embedding(max_text_length, bert_f_dim)

        self.enc_fc_in = MLP(input_dim=bert_f_dim+bert_f_dim+bert_f_dim+bert_f_dim, hidden_dim=bert_f_dim, output_dim=hidden_dim, num_layers=3)

        self.enc_transformer = TransformerWithToken(
                                d_model=hidden_dim,
                                dropout=0.1,
                                nhead=8,
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                normalize_before=False,
                                return_intermediate_dec=False,
                            )
        self.fc_out_disc = nn.Linear(hidden_dim, 1)

        ####################################
        # decoder
        ####################################
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, hidden_dim))
        self.dec_fc_in = nn.Linear(hidden_dim+hidden_dim, hidden_dim)

        te = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=2048)
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=6)

        self.bbox_embed = nn.Linear(hidden_dim, 4)
        self.fc_out_cls = nn.Linear(hidden_dim, num_bbox_labels)

        # text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = im_f_dim
        decoder_config.num_hidden_layers = bert_num_decoder_layers
        decoder_config.num_attention_heads = bert_num_heads
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # text length decoder
        self.fc_text_len_rec = nn.Linear(hidden_dim, max_text_length)

        # image decoder
        #self.im_decoder = Decoder(z_dim=hidden_dim, w_dim=im_f_dim, channel_max=im_f_dim, channel_base=8192, img_channels=img_channels, img_resolution=256, use_noise=False, num_fp16_res=0, conv_clamp=None, fused_modconv_default=False)
        self.bg_decoder = Decoder(z_dim=hidden_dim, w_dim=im_f_dim, channel_max=im_f_dim, channel_base=8192, img_channels=img_channels, img_resolution=background_size, use_noise=False, num_fp16_res=0, conv_clamp=None, fused_modconv_default=False)

        ####################################
        # unconditional discriminator
        ####################################
        self.fc_bbox_uncond = nn.Linear(4, bert_f_dim)
        self.emb_label_uncond = nn.Embedding(num_bbox_labels, bert_f_dim)
        self.enc_fc_in_uncond = MLP(input_dim=bert_f_dim+bert_f_dim, hidden_dim=bert_f_dim, output_dim=hidden_dim, num_layers=3)
        self.enc_transformer_uncond = TransformerWithToken_layoutganpp(d_model=hidden_dim, dim_feedforward=2048, nhead=8, num_layers=6)
        self.fc_out_disc_uncond = nn.Linear(hidden_dim, 1)

        self.pos_token_uncond = nn.Parameter(torch.rand(max_bbox, 1, hidden_dim))
        self.dec_fc_in_uncond = nn.Linear(hidden_dim+hidden_dim, hidden_dim)
        te_uncond = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=2048)
        self.dec_transformer_uncond = nn.TransformerEncoder(te_uncond, num_layers=6)
        self.bbox_embed_uncond = nn.Linear(hidden_dim, 4)
        self.fc_out_cls_uncond = nn.Linear(hidden_dim, num_bbox_labels)

    def forward(self, bbox, bbox_class, bbox_text, bbox_patch, padding_mask, background, c, reconst=False):
        if isinstance(background, (list, torch.Tensor)):
            background = nested_tensor_from_tensor_list(background)
        bg_feat, pos = self.backbone(background)
        bg_feat, mask = bg_feat[-1].decompose()
        assert mask is not None

        B, N, C, H, W = bbox_patch.size()
        b = self.fc_bbox(bbox)
        l = self.emb_label(bbox_class)
        text = self.tokenizer(merge_lists(bbox_text), padding='max_length', truncation=True, max_length=self.max_text_length, return_tensors="pt").to(bbox_class.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_feat = text_output.last_hidden_state[:,0,:].view(B, N, -1)
        #text_len = torch.from_numpy(np.array([float(len(t))/40.0 for t in merge_lists(bbox_text)])).to(bbox_class.device).to(torch.float32).view(B, N, 1)
        text_len = torch.from_numpy(np.array([len(t) for t in merge_lists(bbox_text)])).to(bbox_class.device).to(torch.int64).view(B, N)
        text_len_feat = self.enc_text_len(text_len)
        x = torch.cat([b, l, text_feat, text_len_feat], dim=-1)
        x = torch.relu(self.enc_fc_in(x)).permute(1, 0, 2)
        
        x = self.enc_transformer(src=self.input_proj(bg_feat), mask=mask, pos_embed=pos[-1], tgt=x, tgt_key_padding_mask=padding_mask)[0].transpose(0, 1)
        x0 = x[0]
        logit_disc = self.fc_out_disc(x0).squeeze(-1)

        # unconditional discriminator
        b_uncond = self.fc_bbox_uncond(bbox)
        l_uncond = self.emb_label_uncond(bbox_class)
        x_uncond = torch.cat([b_uncond, l_uncond], dim=-1)
        x_uncond = torch.relu(self.enc_fc_in_uncond(x_uncond)).permute(1, 0, 2)
        x_uncond = self.enc_transformer_uncond(x_uncond, src_key_padding_mask=padding_mask)
        x0_uncond = x_uncond[0]
        logit_disc_uncond = self.fc_out_disc_uncond(x0_uncond).squeeze(-1)

        if not reconst:
            return logit_disc, logit_disc_uncond

        else:
            x = x0.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))

            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            x = x.permute(1, 0, 2)[~padding_mask]

            # bbox_pred: [M, 4]     logit_cls: [M, L]    
            bbox_pred = self.bbox_embed(x).sigmoid()
            logit_cls = self.fc_out_cls(x)

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

            # text length decoder
            text_len_rec = self.fc_text_len_rec(x)
            loss_text_len = F.cross_entropy(text_len_rec, text_len[~padding_mask])

            # image and background decoder
            #im_rec = self.im_decoder(x)
            #im_rec[bbox_patch[~padding_mask]==0.0] = 0.0
            bg_rec = self.bg_decoder(x0)

            # unconditional discriminator
            x_uncond = x0_uncond.unsqueeze(0).expand(N, -1, -1)
            t_uncond = self.pos_token_uncond[:N].expand(-1, B, -1)
            x_uncond = torch.cat([x_uncond, t_uncond], dim=-1)
            x_uncond = torch.relu(self.dec_fc_in_uncond(x_uncond))
            x_uncond = self.dec_transformer_uncond(x_uncond, src_key_padding_mask=padding_mask)
            x_uncond = x_uncond.permute(1, 0, 2)[~padding_mask]
            bbox_pred_uncond = self.bbox_embed_uncond(x_uncond).sigmoid()
            logit_cls_uncond = self.fc_out_cls_uncond(x_uncond)

            return logit_disc, logit_disc_uncond, bbox_pred, logit_cls, loss_lm, loss_text_len, bg_rec, bbox_pred_uncond, logit_cls_uncond
