import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

from metrics.metric_layoutnet import compute_overlap, compute_alignment, generalized_iou_loss

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, bbox_real, bbox_class, bbox_text, bbox_patch, padding_mask, background, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10.0, style_mixing_prob=0, pl_weight=2.0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                 Dreal_bbox_cls_weight=1.0, Dreal_bbox_rec_weight=10.0, Dreal_text_rec_weight=0.01, Dreal_text_len_rec_weight=10.0, Dreal_im_rec_weight=1.0,
                 Ggen_bbox_rec_weight=10.0, Ggen_bbox_gIoU_weight=1.0, Ggen_overlapping_weight=7.0, Ggen_alignment_weight=17.0,
                 Ggen_z_rec_weight=10.0, Ggen_bbox_cls_weight=1.0, Ggen_text_rec_weight=0.01, Ggen_text_len_rec_weight=10.0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.Dreal_bbox_cls_weight = Dreal_bbox_cls_weight
        self.Dreal_bbox_rec_weight = Dreal_bbox_rec_weight
        self.Dreal_text_rec_weight = Dreal_text_rec_weight
        self.Dreal_text_len_rec_weight = Dreal_text_len_rec_weight
        self.Dreal_im_rec_weight = Dreal_im_rec_weight
        self.Ggen_bbox_rec_weight = Ggen_bbox_rec_weight
        self.Ggen_bbox_gIoU_weight = Ggen_bbox_gIoU_weight
        self.Ggen_overlapping_weight = Ggen_overlapping_weight
        self.Ggen_alignment_weight = Ggen_alignment_weight
        self.Ggen_z_rec_weight = Ggen_z_rec_weight
        self.Ggen_bbox_cls_weight = Ggen_bbox_cls_weight
        self.Ggen_text_rec_weight = Ggen_text_rec_weight
        self.Ggen_text_len_rec_weight = Ggen_text_len_rec_weight

    def run_G(self, z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, c, reconst=False, update_emas=False):
        if not reconst:
            bbox_fake = self.G(z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, c)
            return bbox_fake
        bbox_fake, loss_z, bbox_cls_logits, loss_lm, loss_text_len = self.G(z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, c, reconst)
        return bbox_fake, loss_z, bbox_cls_logits, loss_lm, loss_text_len

    def run_D(self, bbox, bbox_class, bbox_text, bbox_patch, padding_mask, background, c, reconst=False, blur_sigma=0, update_emas=False):
        if not reconst:
            logits, logits_uncond = self.D(bbox, bbox_class, bbox_text, bbox_patch, padding_mask, background, c)
            return logits, logits_uncond
        logits, logits_uncond, bbox_rec, bbox_cls_logits, loss_lm, loss_text_len, bg_rec, bbox_rec_uncond, bbox_cls_logits_uncond = self.D(bbox, bbox_class, bbox_text, bbox_patch, padding_mask, background, c, reconst)
        return logits, logits_uncond, bbox_rec, bbox_cls_logits, loss_lm, loss_text_len, bg_rec, bbox_rec_uncond, bbox_cls_logits_uncond

    def accumulate_gradients(self, phase, bbox_real, bbox_class, bbox_text, bbox_patch, padding_mask, background, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                bbox_fake, loss_z, bbox_cls_logits, loss_lm, loss_text_len = self.run_G(gen_z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, gen_c, reconst=True)
                gen_logits, gen_logits_uncond = self.run_D(bbox_fake, bbox_class, bbox_text, bbox_patch, padding_mask, background, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Ggen = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss_Ggen', loss_Ggen)
                loss_Ggen_uncond = torch.nn.functional.softplus(-gen_logits_uncond) # -log(sigmoid(gen_logits_uncond))
                training_stats.report('Loss/G/loss_Ggen_uncond', loss_Ggen_uncond)
                loss_Ggen_bbox_rec = torch.nn.functional.mse_loss(bbox_fake[~padding_mask], bbox_real[~padding_mask])
                loss_Ggen_bbox_rec *= self.Ggen_bbox_rec_weight
                training_stats.report('Loss/G/loss_Ggen_bbox_rec', loss_Ggen_bbox_rec)
                loss_Ggen_bbox_gIoU = generalized_iou_loss(bbox_fake[~padding_mask], bbox_real[~padding_mask])
                loss_Ggen_bbox_gIoU *= self.Ggen_bbox_gIoU_weight
                training_stats.report('Loss/G/loss_Ggen_bbox_gIoU', loss_Ggen_bbox_gIoU)
                loss_Ggen_overlapping = compute_overlap(bbox_fake, ~padding_mask)
                loss_Ggen_overlapping *= self.Ggen_overlapping_weight
                training_stats.report('Loss/G/loss_Ggen_overlapping', loss_Ggen_overlapping)
                loss_Ggen_alignment = compute_alignment(bbox_fake, ~padding_mask)
                loss_Ggen_alignment *= self.Ggen_alignment_weight
                training_stats.report('Loss/G/loss_Ggen_alignment', loss_Ggen_alignment)
                loss_Ggen_z_rec = loss_z * self.Ggen_z_rec_weight
                training_stats.report('Loss/G/loss_Ggen_z_rec', loss_Ggen_z_rec)
                loss_Ggen_bbox_cls = torch.nn.functional.cross_entropy(bbox_cls_logits, bbox_class[~padding_mask])
                loss_Ggen_bbox_cls *= self.Ggen_bbox_cls_weight
                training_stats.report('Loss/G/loss_Ggen_bbox_cls', loss_Ggen_bbox_cls)
                loss_Ggen_text_rec = loss_lm * self.Ggen_text_rec_weight
                training_stats.report('Loss/G/loss_Ggen_text_rec', loss_Ggen_text_rec)
                loss_Ggen_text_len_rec = loss_text_len * self.Ggen_text_len_rec_weight
                training_stats.report('Loss/G/loss_Ggen_text_len_rec', loss_Ggen_text_len_rec)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Ggen + loss_Ggen_uncond + loss_Ggen_bbox_rec + loss_Ggen_bbox_gIoU + loss_Ggen_overlapping + loss_Ggen_alignment + loss_Ggen_z_rec + loss_Ggen_bbox_cls + loss_Ggen_text_rec + loss_Ggen_text_len_rec).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_z_temp = gen_z[:batch_size].requires_grad_(True)
                bbox_class_temp = bbox_class[:batch_size]
                bbox_real_temp = bbox_real[:batch_size]
                bbox_text_temp = bbox_text[:batch_size]
                bbox_patch_temp = bbox_patch[:batch_size]
                padding_mask_temp = padding_mask[:batch_size]
                background_temp = background[:batch_size]
                gen_c_temp = gen_c[:batch_size]
                bbox_fake = self.run_G(gen_z_temp, bbox_class_temp, bbox_real_temp, bbox_text_temp, bbox_patch_temp, padding_mask_temp, background_temp, gen_c_temp)
                pl_noise = torch.randn_like(bbox_fake) / float(bbox_fake.shape[2])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(bbox_fake * pl_noise).sum()], inputs=[gen_z_temp], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum([1,2]).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                bbox_fake = self.run_G(gen_z, bbox_class, bbox_real, bbox_text, bbox_patch, padding_mask, background, gen_c, update_emas=True)
                gen_logits, gen_logits_uncond = self.run_D(bbox_fake, bbox_class, bbox_text, bbox_patch, padding_mask, background, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_Dgen', loss_Dgen)
                loss_Dgen_uncond = torch.nn.functional.softplus(gen_logits_uncond) # -log(1 - sigmoid(gen_logits_uncond))
                training_stats.report('Loss/D/loss_Dgen_uncond', loss_Dgen_uncond)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + loss_Dgen_uncond).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                bbox_real_tmp = bbox_real.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits, real_logits_uncond, bbox_rec, bbox_cls_logits, loss_lm, loss_text_len, bg_rec, bbox_rec_uncond, bbox_cls_logits_uncond = self.run_D(bbox_real_tmp, bbox_class, bbox_text, bbox_patch, padding_mask, background, real_c, reconst=True, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_Dreal_uncond = 0
                loss_Dreal_bbox_rec = 0
                loss_Dreal_bbox_cls = 0
                loss_Dreal_text_rec = 0
                #loss_Dreal_im_rec = 0
                loss_Dreal_bg_rec = 0
                loss_Dreal_bbox_rec_uncond = 0
                loss_Dreal_bbox_cls_uncond = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_uncond = torch.nn.functional.softplus(-real_logits_uncond) # -log(sigmoid(real_logits_uncond))
                    loss_Dreal_bbox_rec = torch.nn.functional.mse_loss(bbox_rec, bbox_real_tmp[~padding_mask])
                    loss_Dreal_bbox_rec *= self.Dreal_bbox_rec_weight
                    loss_Dreal_bbox_cls = torch.nn.functional.cross_entropy(bbox_cls_logits, bbox_class[~padding_mask])
                    loss_Dreal_bbox_cls *= self.Dreal_bbox_cls_weight
                    loss_Dreal_text_rec = loss_lm * self.Dreal_text_rec_weight
                    loss_Dreal_text_len_rec = loss_text_len * self.Dreal_text_len_rec_weight
                    bbox_patch_gt = bbox_patch[~padding_mask]
                    #loss_Dreal_im_rec = torch.nn.functional.mse_loss(im_rec, bbox_patch_gt, reduction='none')
                    #loss_Dreal_im_rec = (loss_Dreal_im_rec.sum((1,2,3)) / (bbox_patch_gt!=0.0).to(torch.float32).sum((1,2,3))).mean()
                    #loss_Dreal_im_rec *= self.Dreal_im_rec_weight
                    loss_Dreal_bg_rec = torch.nn.functional.mse_loss(bg_rec, background) 
                    loss_Dreal_bg_rec *= self.Dreal_im_rec_weight
                    loss_Dreal_bbox_rec_uncond = torch.nn.functional.mse_loss(bbox_rec_uncond, bbox_real_tmp[~padding_mask])
                    loss_Dreal_bbox_rec_uncond *= self.Dreal_bbox_rec_weight
                    loss_Dreal_bbox_cls_uncond = torch.nn.functional.cross_entropy(bbox_cls_logits_uncond, bbox_class[~padding_mask])
                    loss_Dreal_bbox_cls_uncond *= self.Dreal_bbox_cls_weight
                    training_stats.report('Loss/D/loss_Dreal', loss_Dreal)
                    training_stats.report('Loss/D/loss_Dreal_uncond', loss_Dreal_uncond)
                    training_stats.report('Loss/D/loss_Dreal_bbox_rec', loss_Dreal_bbox_rec)
                    training_stats.report('Loss/D/loss_Dreal_bbox_cls', loss_Dreal_bbox_cls)
                    training_stats.report('Loss/D/loss_Dreal_text_rec', loss_Dreal_text_rec)
                    training_stats.report('Loss/D/loss_Dreal_text_len_rec', loss_Dreal_text_len_rec)
                    #training_stats.report('Loss/D/loss_Dreal_im_rec', loss_Dreal_im_rec)
                    training_stats.report('Loss/D/loss_Dreal_bg_rec', loss_Dreal_bg_rec)
                    training_stats.report('Loss/D/loss_Dreal_bbox_rec_uncond', loss_Dreal_bbox_rec_uncond)
                    training_stats.report('Loss/D/loss_Dreal_bbox_cls_uncond', loss_Dreal_bbox_cls_uncond)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[bbox_real_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dreal_uncond + loss_Dreal_bbox_rec + loss_Dreal_bbox_cls + loss_Dreal_text_rec + loss_Dreal_text_len_rec + loss_Dreal_bg_rec + loss_Dreal_bbox_rec_uncond + loss_Dreal_bbox_cls_uncond + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
