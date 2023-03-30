# LayoutDETR

### [LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](https://arxiv.org/pdf/2212.09877.pdf)
[Ning Yu](https://ningyu1991.github.io/)\*, [Chia-Chih Chen](https://scholar.google.com/citations?user=0Hr1SOUAAAAJ&hl=en)\*, [Zeyuan Chen](https://www.linkedin.com/in/zeyuan-chen-0253b6141/), [Rui Meng](http://memray.me/), [Gang Wu](https://www.linkedin.com/in/whoisgang/), [Paul Josel](https://www.linkedin.com/in/paul-josel/), [Juan Carlos Niebles](http://www.niebles.net/), [Caiming Xiong](http://cmxiong.com/), [Ran Xu](https://www.linkedin.com/in/ran-x-a2765924/)<br>
arXiv 2023

<img src='fig/teaser.png' width=200>
<img src='fig/framework_architecture.png' width=600>

## Abstract
Graphic layout designs play an essential role in visual communication. Yet handcrafting layout designs is skill-demanding, time-consuming, and non-scalable to batch production. Generative models emerge to make design automation scalable but it remains non-trivial to produce designs that comply with designers' multimodal desires, i.e., constrained by background images and driven by foreground content. We propose *LayoutDETR* that inherits the high quality and realism from generative modeling, while reformulating content-aware requirements as a detection problem: we learn to detect in a background image the reasonable locations, scales, and spatial relations for multimodal foreground elements in a layout. Our solution sets a new state-of-the-art performance for layout generation on public benchmarks and on our newly-curated ad banner dataset. We integrate our solution into a graphical system that facilitates user studies, and show that users prefer our designs over baselines by significant margins.

### Data preprocessing
```
python dataset_tool.py \
--source=data/dataset/ads_banner_collection_manual_3x_mask/raw/manual_json_png_gt_label \
--dest=data/dataset/ads_banner_collection_manual_3x_mask/zip
```
where
- `--source` indicates the source data path, which contains two subdirectories. The `manual_json_png_gt_label` subdirectory contains a set of `*.png` files representing well-designed images with foreground elements superimposed on the background. It also correspondingly contains a set of `*.json` files with the same file names as of `*.png`, representing the layout ground truth of foreground elements of each well-designed image. Each `*.json` file contains a set of bounding box annotations in the form of `[cy, cx, height, width]`, their label annotations, and their text contents if any. The `manual_LaMa_3x_stringOnly_inpainted_background_images` subdirectory correspondingly contains a set of `*.png` files representing the background-only images of the well-designed images. The subregions that were superimposed by foreground elements have been inpainted by the [LaMa technique](https://github.com/saic-mdal/lama).
- `--dest` indicates the preprocessed data path containing two files: `train.zip` and `val.zip` which are 9:1 splitted from the source data.

### Launch training
```
python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=50.0 --bbox-rec-weight=500.0 --text-rec-weight=0.1 --text-len-rec-weight=2.0 --im-rec-weight=0.5 \
--bbox-giou-weight=4.0 --overlapping-weight=7.0 --alignment-weight=17.0 --z-rec-weight=5.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-encoder-layers=12 --bert-num-decoder-layers=2 \
--background-size=256 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_manual_3x_mask/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z
```
where
- `--data` indicates the preprocessed training data path.
- `--outdir` indicates the output path of model checkpoints, result snapshots, config record file, log file, etc.
- `--metrics` indicates the evaluation metrics measured for each model checkpoint during training, which can include image FID, layout FID, overlap penalty, misalignment penalty, layout-wise IoU, and layout-wise DocSim, etc. See more metric options in `metrics/`.
- See the definitions and default settings of the other arguments in `train.py`.

### Predict Bounding Boxes
```
python gen_single_sample_API.py \
--seeds=0-4 \
--network=/export/share/ning/projects/webpage_generation/stylegan3_detr_genRec_uncondDis_fixedTextEncoder_unifiedNoise_textNoImageCond_backgroundCond_paddingImageInput_CNN_overlapping_alignment_losses_D_LM_D_visualDecoder/training-runs/layoutganpp/ads_banner_collection_manual_3x_mask/00001-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-008800.pkl \
--bg=/export/share/zeyuan/dataset/content_generation/samples/lumber_Liquidators_living_room.jpg \
--bg-preprocessing=256 \
--strings='10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW|Excludes limited time deals and clearance' --string-labels='header|body text|button|disclaimer / footnote' \
--outdir=API_generated_single_samples/test --out-jittering-strength=0.2 --out-postprocessing=horizontal_center_aligned
```
where
- `--seeds=XX-YY` indicates using different random seeds from range(XX, YY) to generate layouts.
- `--network` indicates the path of the well-trained generator .pkl file.
- `--bg` indicates the path of the provided background image .png file.
- `--bg-preprocessing` indicates the preprocessing operation to the background image. The default is `none`, meaning no preprocessing.
- `--strings` indicates the ads text strings, the bboxes of which will be generated on top of the background image. Multiple (<10) strings are separated by `|`.
- `--string-labels` indicates the ads text string labels, selected from {`header`, `pre-header`, `post-header`, `body text`, `disclaimer / footnote`, `button`, `callout`, `logo`}. Multiple (<10) strings are separated by `|`.
- `--outdir` indicates the output directory, where there are two subdirectories containing N generated layouts of bboxes on top of the background image. N equals to the number of random seeds. Each subdirectory sorts the generated layouts by either the overlapping penalty (the smaller the less overlapping artifacts) or the alignment penalty (the smaller the stronger left/right/top/bottom/center alignment).
- `--out-jittering-strength` indicates the strength of randomly jitterring, in the range of 0.0-1.0 (0-100%), to the output bbox parameters, so as to diversify layout generation. The default is 0.0, meaning no jittering.
- `--out-postprocessing` indicates the postprocessing operation to the output bbox parameters so as to guarantee alignment and remove overlapping. The operation can be selected from {`none`, `horizontal_center_aligned`, `horizontal_left_aligned`}. The default is `none`, meaning no postprocessing.
- The values of generated bbox parameters [cy, cx, h, w] can be read from the variable `bbox_fake` (in the shape of BxNx4, B=1, N=#strings in one ads) in `gen_single_sample_API.py`.