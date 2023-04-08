import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:       {c.run_dir}')
    print(f'Number of GPUs:         {c.num_gpus}')
    print(f'Batch size:             {c.batch_size} images')
    print(f'Training duration:      {c.total_kimg} kimg')
    print(f'Training dataset path:  {c.training_set_kwargs.path}')
    print(f'Training dataset size:  {c.training_set_kwargs.max_size} images')
    #print(f'Training dataset height:{c.training_set_kwargs.height}')
    #print(f'Training dataset width: {c.training_set_kwargs.width}')
    print(f'Training dataset labels:{c.training_set_kwargs.use_labels}')
    #print(f'Training dataset x-flips:     {c.training_set_kwargs.xflip}')
    print(f'Validation dataset path:{c.validation_set_kwargs.path}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    #torch.multiprocessing.set_start_method('fork')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, background_size):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_layoutganpp.LayoutDataset', path=data, use_labels=False, max_size=None, xflip=False, background_size=background_size)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        #dataset_kwargs.height = dataset_obj.image_shape[1] # Be explicit about image height.
        #dataset_kwargs.width = dataset_obj.image_shape[2] # Be explicit about image width.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP]',                          type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--pl-weight',    help='Path length regularization weight', metavar='FLOAT',      type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--bbox-cls-weight', help='Discriminator/generator bottleneck bbox classification weight', metavar='FLOAT', type=click.FloatRange(min=0), default=50.0, show_default=True)
@click.option('--bbox-rec-weight', help='Discriminator/generator bottleneck bbox reconstruction weight', metavar='FLOAT', type=click.FloatRange(min=0), default=500.0, show_default=True)
@click.option('--text-rec-weight', help='Discriminator/generator bottleneck text reconstruction weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--text-len-rec-weight', help='Discriminator/generator bottleneck text length reconstruction weight', metavar='FLOAT', type=click.FloatRange(min=0), default=2.0, show_default=True)
@click.option('--im-rec-weight', help='Discriminator bottleneck image reconstruction weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--bbox-giou-weight', help='Generator bbox supervised reconstruction weight', metavar='FLOAT', type=click.FloatRange(min=0), default=4.0, show_default=True)
@click.option('--overlapping-weight', help='Generator bbox overlapping penalty weight', metavar='FLOAT', type=click.FloatRange(min=0), default=7.0, show_default=True)
@click.option('--alignment-weight', help='Generator bbox alignment penalty weight', metavar='FLOAT', type=click.FloatRange(min=0), default=17.0, show_default=True)
@click.option('--z-rec-weight', help='Generator noise reconstruction weight', metavar='FLOAT', type=click.FloatRange(min=0), default=5.0, show_default=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--resume-kimg',  help='Resume kimg index from given network pickle', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=1e-5, show_default=True)
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=1e-5, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=0), default=4, show_default=True)

# Layoutganpp hyperparameters (arg keys must be in lower case!!!).
@click.option('--z-dim',        help='G latent input dimention', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--g-f-dim',      help='G intermediate feature dimention', metavar='INT',         type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--g-num-heads',  help='G number of attention heads', metavar='INT',              type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--g-num-layers', help='G number of attention layers', metavar='INT',             type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--d-f-dim',      help='D intermediate feature dimention', metavar='INT',         type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--d-num-heads',  help='D number of attention heads', metavar='INT',              type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--d-num-layers', help='D number of attention layers', metavar='INT',             type=click.IntRange(min=1), default=8, show_default=True)

# Layoutganpp BERT text encoder/decoder hyperparameters (arg keys must be in lower case!!!).
@click.option('--bert-f-dim',    help='BERT intermediate feature dimention', metavar='INT',     type=click.IntRange(min=1), default=768, show_default=True)
@click.option('--bert-num-heads',help='BERT number of attention heads', metavar='INT',          type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--bert-num-encoder-layers', help='BERT encoder number of attention layers', metavar='INT', type=click.IntRange(min=1), default=12, show_default=True)
@click.option('--bert-num-decoder-layers', help='BERT decoder number of attention layers', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)

# Layoutganpp image encoder/decoder hyperparameters (arg keys must be in lower case!!!).
@click.option('--background-size', help='Background image resolution for encoder/decoder training', metavar='INT', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--im-f-dim',     help='Image encoder/decoder intermediate feature dimention', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=8, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name='training.networks_detr.Generator')
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_detr.Discriminator')
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, background_size=opts.background_size)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Validation set.
    c.validation_set_kwargs, _ = init_dataset_kwargs(data=opts.data.replace('train.zip', 'val.zip'), background_size=opts.background_size)
    if opts.cond and not c.validation_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.validation_set_kwargs.use_labels = opts.cond
    c.validation_set_kwargs.xflip = False

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = opts.snap
    c.network_snapshot_ticks = opts.snap * 10
    c.random_seed = c.training_set_kwargs.random_seed = c.validation_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    #if c.D_kwargs.epilogue_kwargs.mbstd_group_size > 0 and c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
    #    raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.z_dim = opts.z_dim
    c.G_kwargs.f_dim = opts.g_f_dim
    c.G_kwargs.num_heads = opts.g_num_heads
    c.G_kwargs.num_layers = opts.g_num_layers
    c.D_kwargs.f_dim = opts.d_f_dim
    c.D_kwargs.num_heads = opts.d_num_heads
    c.D_kwargs.num_layers = opts.d_num_layers
    c.G_kwargs.bert_f_dim = c.D_kwargs.bert_f_dim = opts.bert_f_dim
    c.G_kwargs.bert_num_heads = c.D_kwargs.bert_num_heads = opts.bert_num_heads
    c.G_kwargs.bert_num_encoder_layers = c.D_kwargs.bert_num_encoder_layers = opts.bert_num_encoder_layers
    c.G_kwargs.bert_num_decoder_layers = c.D_kwargs.bert_num_decoder_layers = opts.bert_num_decoder_layers
    c.G_kwargs.im_f_dim = c.D_kwargs.im_f_dim = opts.im_f_dim
    c.loss_kwargs.pl_weight = opts.pl_weight # Enable path length regularization.
    c.loss_kwargs.Dreal_bbox_cls_weight = opts.bbox_cls_weight
    c.loss_kwargs.Ggen_bbox_cls_weight = opts.bbox_cls_weight
    c.loss_kwargs.Dreal_bbox_rec_weight = opts.bbox_rec_weight
    c.loss_kwargs.Ggen_bbox_rec_weight = opts.bbox_rec_weight / 5.0
    c.loss_kwargs.Dreal_text_rec_weight = opts.text_rec_weight
    c.loss_kwargs.Ggen_text_rec_weight = opts.text_rec_weight * 10.0
    c.loss_kwargs.Dreal_text_len_rec_weight = opts.text_len_rec_weight
    c.loss_kwargs.Ggen_text_len_rec_weight = opts.text_len_rec_weight / 2.0
    c.loss_kwargs.Dreal_im_rec_weight = opts.im_rec_weight
    c.loss_kwargs.Ggen_bbox_gIoU_weight = opts.bbox_giou_weight
    c.loss_kwargs.Ggen_overlapping_weight = opts.overlapping_weight
    c.loss_kwargs.Ggen_alignment_weight = opts.alignment_weight
    c.loss_kwargs.Ggen_z_rec_weight = opts.z_rec_weight
    c.G_reg_interval = 4 # Enable lazy regularization for G.
    c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.resume_kimg = opts.resume_kimg
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    #if opts.fp32:
    #    c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #    c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'layoutganpp-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-pl{c.loss_kwargs.pl_weight:.3f}-gamma{c.loss_kwargs.r1_gamma:.3f}-overlapping{c.loss_kwargs.Ggen_overlapping_weight:.0f}-alignment{c.loss_kwargs.Ggen_alignment_weight:.0f}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
