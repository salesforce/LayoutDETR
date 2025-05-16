# LayoutDETR

### [LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](https://arxiv.org/pdf/2212.09877.pdf)
[Ning Yu](https://ningyu1991.github.io/), [Chia-Chih Chen](https://scholar.google.com/citations?user=0Hr1SOUAAAAJ&hl=en), [Zeyuan Chen](https://www.linkedin.com/in/zeyuan-chen-0253b6141/), [Rui Meng](http://memray.me/)<br>[Gang Wu](https://www.linkedin.com/in/whoisgang/), [Paul Josel](https://www.linkedin.com/in/paul-josel/), [Juan Carlos Niebles](http://www.niebles.net/), [Caiming Xiong](http://cmxiong.com/), [Ran Xu](https://www.linkedin.com/in/ran-x-a2765924/)<br>

Salesforce Research

arXiv 2023

### [paper](https://arxiv.org/pdf/2212.09877.pdf) | [project page](https://ningyu1991.github.io/projects/LayoutDETR.html)

<pre><img src='assets/teaser.png' width=200>	<img src='assets/framework_architecture.png' width=400></pre>
<img src='assets/samples_ads_cgl.jpg' width=700></pre>

## Abstract
Graphic layout designs play an essential role in visual communication. Yet handcrafting layout designs is skill-demanding, time-consuming, and non-scalable to batch production. Generative models emerge to make design automation scalable but it remains non-trivial to produce designs that comply with designers' multimodal desires, i.e., constrained by background images and driven by foreground content. We propose *LayoutDETR* that inherits the high quality and realism from generative modeling, while reformulating content-aware requirements as a detection problem: we learn to detect in a background image the reasonable locations, scales, and spatial relations for multimodal foreground elements in a layout. Our solution sets a new state-of-the-art performance for layout generation on public benchmarks and on our newly-curated ad banner dataset. We integrate our solution into a graphical system that facilitates user studies, and show that users prefer our designs over baselines by significant margins.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 11.3
- To install conda virtual environment, run
	```
	conda env create -f environment.yaml
	conda activate layoutdetr
	```
- For training, download [Up-DETR pretrained weights](https://drive.google.com/file/d/1JhL1uwNJCaxMrIUx7UzQ3CMCHqmZpCnn/view?usp=sharing) to `pretrained/`.
- For inference and layout generation in the wild, build Chrome-based text rendering environment by running
	```
	apt-get update
	cp chromedriver /usr/bin/chromedriver
	ln -fs /usr/share/zoneinfo/America/Los_Angelos /etc/localtime
	DEBIAN_FRONTEND=noninteractive apt --assume-yes install ./google-chrome-stable_current_amd64.deb
	```

## Data preprocessing
[Our ad banner dataset](https://drive.google.com/file/d/1T09t4dX7zQ7J-8KxtJv1RkyjRNdilD1m/view?usp=sharing
) (14.7GB, 7,672 samples). Part of the source images are filtered from [Pitt Image Ads Dataset](https://people.cs.pitt.edu/~kovashka/ads/readme_images.txt) and the others are crawled from Google image search engine with a variety of retailer brands as keywords. Download our dataset and unzip to `data/` which contains three subdirectories:
- `png_json_gt/` subdirectory contains:
	- `*.png` files representing well-designed images with foreground elements superimposed on the background.
	- Corresponding `*.json` files with the same file names as of `*.png`, representing the layout ground truth of foreground elements of each well-designed image. Each `*.json` file contains:
		- `xyxy_word_fit` key: A set of bounding box annotations in the form of `[cy, cx, height, width]`, detected by our [Salesforce Einstein OCR](https://help.salesforce.com/s/articleView?id=release-notes.rn_einstein_vision_ocr_pdf_support.htm&release=230&type=5).
		- `str` key: Their text contents if any, also recognized by our [Salesforce Einstein OCR](https://help.salesforce.com/s/articleView?id=release-notes.rn_einstein_vision_ocr_pdf_support.htm&release=230&type=5).
		- `label` key: Their element categories annotated manually through [Amazon Mechanical Turk](https://www.mturk.com/). The interesting categories include {`header`, `pre-header`, `post-header`, `body text`, `disclaimer / footnote`, `button`, `callout`, `logo`}.
- `1x_inpainted_background_png/` subdirectory correspondingly contains a set of `*_inpainted.png` files representing the background-only images of the well-designed images. The subregions that were superimposed by foreground elements have been inpainted by the [LaMa technique](https://github.com/saic-mdal/lama). These background images should be used for inference or evaluation only, **not for training**.
- `3x_inpainted_background_png/` subdirectory also correspondingly contains a set of `*_inpainted.png` files representing the background-only images of the well-designed images. There are 2x extra random subregions also inpainted, which aim at avoiding generator being overfitted to inpainted subregions if we inpaint only ground truth layouts. The augmented inpainting subregions serve as false postive which are inpainted but are not ground truth layouts. We use these background images for training.

To preprocess the dataset that are efficient for training, run
```
python dataset_tool.py \
--source=data/ads_banner_dataset/png_json_gt \
--dest=data/ads_banner_dataset/zip_3x_inpainted \
--inpaint-aug
```
where
- `--source` indicates the source data direcotry path where you downloaded the raw dataset.
- `--dest` indicates the preprocessed data direcotry path containing two files: `train.zip` and `val.zip` which are 9:1 splitted from the source data.
- `inpaint-aug` indicates using `3x_inpainted_background_png/` with extra inpainting on background instead of using `1x_inpainted_background_png/`. Use this argument when preprocessing training data.

## Training
```
python train.py --gpus=8 --batch=16 \
--data=data/ads_banner_dataset/zip_3x_inpainted/train.zip \
--outdir=training-runs \
--metrics=layout_fid50k_train,layout_fid50k_val,fid50k_train,fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val
```
where
- `--batch` indicates the **total batch size** on all the GPUs.
- `--data` indicates the preprocessed training data .zip file path.
- `--outdir` indicates the output direcotry path of model checkpoints, result snapshots, config record file, log file, etc.
- `--metrics` indicates the evaluation metrics measured for each model checkpoint during training, which can include layout FID, image FID, overlap penalty, misalignment penalty, layout-wise IoU, and layout-wise DocSim, etc. See more metric options in `metrics/metric_main.py`.
- See the definitions and default settings of the other arguments in `train.py`.

## Evaluation
Download the **well-trained LayoutDETR model** on our ad banner dataset from [here](https://drive.google.com/file/d/1iaKATX2Id9JnqDunDytVIK5l9HO0a0w-/view?usp=drive_link) (2.7GB) to `checkpoints/`.
```
python evaluate.py --gpus=8 --batch=16 \
--data=data/ads_banner_dataset/zip_1x_inpainted/val.zip \
--outdir=evaluation \
--ckpt=checkpoints/layoutdetr_ad_banner.pkl \
--metrics=layout_fid50k_val,fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,rendering_val
```
where
- `--ckpt` indicates the path of the well-trained generator .pkl file.
- `--metrics=rendering_val` indicates to render texts on background images given generated layouts.

## Layout generation in the wild
```
python generate.py \
--ckpt=checkpoints/layoutdetr_ad_banner.pkl \
--bg='examples/Lumber 2 [header]EVERYTHING 10% OFF[body text]Friends & Family Savings Event[button]SHOP NOW[disclaimer]CODE FRIEND10.jpg' \
--bg-preprocessing=256 \
--strings='EVERYTHING 10% OFF|Friends & Family Savings Event|SHOP NOW|CODE FRIEND10' \
--string-labels='header|body text|button|disclaimer / footnote' \
--outfile='examples/output/Lumber 2' \
--out-postprocessing=horizontal_center_aligned
```
where
- `--ckpt` indicates the path of the well-trained generator .pkl file.
- `--bg` indicates the provided background image file path.
- `--bg-preprocessing` indicates the preprocessing operation to the background image. The default is `none`, meaning no preprocessing.
- `--strings` indicates the ads text strings, the bboxes of which will be generated on top of the background image. Multiple (<10) strings are separated by `|`.
- `--string-labels` indicates the ads text string labels, selected from {`header`, `pre-header`, `post-header`, `body text`, `disclaimer / footnote`, `button`, `callout`, `logo`}. Multiple (<10) strings are separated by `|`.
- `--outfile` indicates the output file path and name (without extension).
- `--out-postprocessing` indicates the postprocessing operation to the output bbox parameters so as to guarantee alignment and remove overlapping. The operation can be selected from {`none`, `horizontal_center_aligned`, `horizontal_left_aligned`}. The default is `none`, meaning no postprocessing.
- The values of generated bbox parameters [cy, cx, h, w] can be read from the variable `bbox_fake` (in the shape of BxNx4, where B=1 and N=#strings in one ads) in `generate.py`.

## Citation
  ```
  @inproceedings{yu2024layoutdetr,
	  title={LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer},
	  author={Yu, Ning and Chen, Chia-Chih and Chen, Zeyuan and Meng, Rui and Wu, Gang and Josel, Paul and Niebles, Juan Carlos and Xiong, Caiming and Xu, Ran},
	  booktitle={European Conference on Computer Vision (ECCV)},
	  year={2024}
  }
  ```

## Acknowledgement
- We thank Abigail Kutruff, [Brian Brechbuhl](https://www.linkedin.com/in/brianbrechbuhl), [Elham Etemad](https://ca.linkedin.com/in/elhametemad), and [Amrutha Krishnan](https://www.linkedin.com/in/amruthakrishnan) from Salesforce for constructive advice.
- We express gratitudes to the [StyleGAN3](https://github.com/NVlabs/stylegan3), [LayoutGAN++](https://github.com/ktrk115/const_layout), [DETR](https://github.com/facebookresearch/detr), [Up-DETR](https://github.com/dddzg/up-detr), and [BLIP](https://github.com/salesforce/BLIP), as our code was modified from their repositories.
- We also acknowledge the data contribution of [Pitt Image Ads Dataset](https://people.cs.pitt.edu/~kovashka/ads/) and technical contribution of [LaMa](https://github.com/saic-mdal/lama).
