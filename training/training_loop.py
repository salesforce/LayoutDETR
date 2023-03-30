import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

from util import save_image, save_real_image, save_real_image_with_background

from training.networks_layoutganpp import split_list

#----------------------------------------------------------------------------

def setup_snapshot(training_set, batch_size, batch_gpu, z_dim, device, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(batch_size)]

    # Load data.
    samples, labels = zip(*[training_set[i] for i in grid_indices])
    W_page = [sample['W_page'] for sample in samples]
    H_page = [sample['H_page'] for sample in samples]
    bbox_real = torch.from_numpy(np.stack((sample['bboxes'] for sample in samples), axis=0)).to(device).to(torch.float32).split(batch_gpu)
    bbox_class = torch.from_numpy(np.stack((sample['labels'] for sample in samples), axis=0)).to(device).to(torch.int64).split(batch_gpu)
    bbox_text = [sample['texts'] for sample in samples]
    bbox_text = split_list(bbox_text, batch_gpu)
    bbox_patch = torch.from_numpy(np.stack((sample['patches'] for sample in samples), axis=0)).to(device).to(torch.float32).split(batch_gpu)
    bbox_patch_orig = torch.from_numpy(np.stack((sample['patches_orig'] for sample in samples), axis=0)).to(device).to(torch.float32).split(batch_gpu)
    bbox_patch_mask = torch.from_numpy(np.stack((sample['patch_masks'] for sample in samples), axis=0)).to(device).to(torch.float32).split(batch_gpu)
    mask = torch.from_numpy(np.stack((sample['mask'] for sample in samples), axis=0)).to(device).to(torch.bool)
    padding_mask = (~mask).split(batch_gpu)
    background = torch.from_numpy(np.stack((sample['background'] for sample in samples), axis=0)).to(device).to(torch.float32).split(batch_gpu)
    background_orig = torch.from_numpy(np.stack((sample['background_orig'] for sample in samples), axis=0)).to(device).to(torch.float32).split(batch_gpu)
    z = torch.randn([batch_size, bbox_class[0].shape[1], z_dim], dtype=torch.float32, device=device).split(batch_gpu)
    c = torch.from_numpy(np.stack(labels, axis=0)).to(device).split(batch_gpu)
    return W_page, H_page, bbox_real, bbox_class, bbox_text, bbox_patch, bbox_patch_orig, bbox_patch_mask, padding_mask, background, background_orig, z, c

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    validation_set_kwargs   = {},       # Options for validation set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    validation_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs) # subclass of training.dataset.Dataset
    if rank == 0:
        print()
        print('Num training images: ', len(training_set))
        print('Num validation images: ', len(validation_set))
        print('Bbox patch shape:', training_set.patch_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(num_bbox_labels=training_set.num_bbox_labels,
                         img_channels=training_set.num_channels,
                         img_height=training_set.height,
                         img_width=training_set.width,
                         background_size=training_set.background_size_for_training,
                         c_dim=training_set.label_dim)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.   
    G.load_state_dict(torch.load('pretrained/up-detr-pre-training-60ep-imagenet.pth')['model'], strict=False)
    D.load_state_dict(torch.load('pretrained/up-detr-pre-training-60ep-imagenet.pth')['model'], strict=False)
    G_ema.load_state_dict(torch.load('pretrained/up-detr-pre-training-60ep-imagenet.pth')['model'], strict=False)
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        W_page_fixed_train, H_page_fixed_train, bbox_real_fixed_train, bbox_class_fixed_train, bbox_text_fixed_train, bbox_patch_fixed_train, bbox_patch_orig_fixed_train, bbox_patch_mask_fixed_train, padding_mask_fixed_train, background_fixed_train, background_orig_fixed_train, z_fixed_train, c_fixed_train = setup_snapshot(training_set, batch_size, batch_gpu, G.z_dim, device)
        bbox_real_temp = bbox_real_fixed_train[0].clone().detach().to(device)
        bbox_class_temp = bbox_class_fixed_train[0].clone().detach().to(device)
        bbox_text_temp = list(bbox_text_fixed_train[0])
        bbox_patch_temp = bbox_patch_fixed_train[0].clone().detach().to(device)
        padding_mask_temp = padding_mask_fixed_train[0].clone().detach().to(device)
        background_temp = background_fixed_train[0].clone().detach().to(device)
        z_temp = torch.empty(z_fixed_train[0].shape, dtype=z_fixed_train[0].dtype, device=device)
        c_temp = torch.empty(c_fixed_train[0].shape, dtype=c_fixed_train[0].dtype, device=device)
        bbox_fake_temp, _, _, _, _ = misc.print_module_summary(G, [z_temp, bbox_class_temp, bbox_real_temp, bbox_text_temp, bbox_patch_temp, padding_mask_temp, background_temp, c_temp, True])
        misc.print_module_summary(D, [bbox_fake_temp, bbox_class_temp, bbox_text_temp, bbox_patch_temp, padding_mask_temp, background_temp, c_temp, True])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        save_image(torch.cat(bbox_real_fixed_train), torch.cat(bbox_class_fixed_train), ~torch.cat(padding_mask_fixed_train), training_set.colors,
                    os.path.join(run_dir, 'train_layouts_real.png'), W_page_fixed_train, H_page_fixed_train)
        save_real_image(torch.cat(bbox_real_fixed_train), torch.cat(bbox_real_fixed_train), torch.cat(bbox_patch_orig_fixed_train), ~torch.cat(padding_mask_fixed_train),
                        os.path.join(run_dir, 'train_images_real.png'), W_page_fixed_train, H_page_fixed_train)
        save_real_image_with_background(torch.cat(bbox_real_fixed_train), torch.cat(bbox_real_fixed_train), torch.cat(bbox_patch_orig_fixed_train), ~torch.cat(padding_mask_fixed_train), torch.cat(background_orig_fixed_train),
                                        os.path.join(run_dir, 'train_images_with_background_real.png'), W_page_fixed_train, H_page_fixed_train)
        W_page_fixed_val, H_page_fixed_val, bbox_real_fixed_val, bbox_class_fixed_val, bbox_text_fixed_val, bbox_patch_fixed_val, bbox_patch_orig_fixed_val, bbox_patch_mask_fixed_val, padding_mask_fixed_val, background_fixed_val, background_orig_fixed_val, z_fixed_val, c_fixed_val = setup_snapshot(validation_set, batch_size, batch_gpu, G.z_dim, device)
        save_image(torch.cat(bbox_real_fixed_val), torch.cat(bbox_class_fixed_val), ~torch.cat(padding_mask_fixed_val), training_set.colors,
                    os.path.join(run_dir, 'val_layouts_real.png'), W_page_fixed_val, H_page_fixed_val)
        save_real_image(torch.cat(bbox_real_fixed_val), torch.cat(bbox_real_fixed_val), torch.cat(bbox_patch_orig_fixed_val), ~torch.cat(padding_mask_fixed_val),
                        os.path.join(run_dir, 'val_images_real.png'), W_page_fixed_val, H_page_fixed_val)
        save_real_image_with_background(torch.cat(bbox_real_fixed_val), torch.cat(bbox_real_fixed_val), torch.cat(bbox_patch_orig_fixed_val), ~torch.cat(padding_mask_fixed_val), torch.cat(background_orig_fixed_val),
                                        os.path.join(run_dir, 'val_images_with_background_real.png'), W_page_fixed_val, H_page_fixed_val)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_samples, phase_real_c = next(training_set_iterator)
            phase_bbox_real = phase_samples['bboxes'].to(device).to(torch.float32).split(batch_gpu)
            phase_bbox_class = phase_samples['labels'].to(device).to(torch.int64).split(batch_gpu)
            phase_samples_text = list(map(list, zip(*(phase_samples['texts'])))) # have to transpose the list of lists of texts
            phase_bbox_text = split_list(phase_samples_text, batch_gpu)
            phase_bbox_patch = phase_samples['patches'].to(device).to(torch.float32).split(batch_gpu)
            phase_mask = phase_samples['mask'].to(device).to(torch.bool)
            phase_padding_mask = (~phase_mask).split(batch_gpu)
            phase_background = phase_samples['background'].to(device).to(torch.float32).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)

            all_gen_z = torch.randn([len(phases) * batch_size, phase_bbox_class[0].shape[1], G.z_dim], dtype=torch.float32, device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            phase.module.text_encoder.requires_grad_(False)
            #phase.module.backbone.requires_grad_(False)
            for bbox_real, bbox_class, bbox_text, bbox_patch, padding_mask, background, real_c, gen_z, gen_c in zip(phase_bbox_real, phase_bbox_class, phase_bbox_text,
                                                                                                                         phase_bbox_patch, phase_padding_mask, phase_background,
                                                                                                                         phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, 
                                          bbox_real=bbox_real,
                                          bbox_class=bbox_class,
                                          bbox_text=bbox_text,
                                          bbox_patch=bbox_patch,
                                          padding_mask=padding_mask,
                                          background=background, 
                                          real_c=real_c, 
                                          gen_z=gen_z, 
                                          gen_c=gen_c, 
                                          gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            bbox_fake_fixed_train = []
            for z_fixed_temp, bbox_class_fixed_temp, bbox_real_fixed_temp, bbox_text_fixed_temp, bbox_patch_fixed_temp, padding_mask_fixed_temp, background_fixed_temp, c_fixed_temp in zip(z_fixed_train, bbox_class_fixed_train, bbox_real_fixed_train, bbox_text_fixed_train, bbox_patch_fixed_train, padding_mask_fixed_train, background_fixed_train, c_fixed_train):
                bbox_fake_fixed_temp = G_ema(z_fixed_temp, bbox_class_fixed_temp, bbox_real_fixed_temp, bbox_text_fixed_temp, bbox_patch_fixed_temp, padding_mask_fixed_temp, background_fixed_temp, c_fixed_temp)
                bbox_fake_fixed_train.append(bbox_fake_fixed_temp.clone())
            save_image(torch.cat(bbox_fake_fixed_train), torch.cat(bbox_class_fixed_train), ~torch.cat(padding_mask_fixed_train), training_set.colors,
                        os.path.join(run_dir, f'train_layouts_fake_{cur_nimg//1000:06d}.png'), W_page_fixed_train, H_page_fixed_train)
            save_real_image(torch.cat(bbox_fake_fixed_train), torch.cat(bbox_real_fixed_train), torch.cat(bbox_patch_orig_fixed_train), ~torch.cat(padding_mask_fixed_train),
                            os.path.join(run_dir, f'train_images_fake_{cur_nimg//1000:06d}.png'), W_page_fixed_train, H_page_fixed_train)
            save_real_image_with_background(torch.cat(bbox_fake_fixed_train), torch.cat(bbox_real_fixed_train), torch.cat(bbox_patch_orig_fixed_train), ~torch.cat(padding_mask_fixed_train), torch.cat(background_orig_fixed_train),
                                            os.path.join(run_dir, f'train_images_with_background_fake_{cur_nimg//1000:06d}.png'), W_page_fixed_train, H_page_fixed_train)
            bbox_fake_fixed_val = []
            for z_fixed_temp, bbox_class_fixed_temp, bbox_real_fixed_temp, bbox_text_fixed_temp, bbox_patch_fixed_temp, padding_mask_fixed_temp, background_fixed_temp, c_fixed_temp in zip(z_fixed_val, bbox_class_fixed_val, bbox_real_fixed_val, bbox_text_fixed_val, bbox_patch_fixed_val, padding_mask_fixed_val, background_fixed_val, c_fixed_val):
                bbox_fake_fixed_temp = G_ema(z_fixed_temp, bbox_class_fixed_temp, bbox_real_fixed_temp, bbox_text_fixed_temp, bbox_patch_fixed_temp, padding_mask_fixed_temp, background_fixed_temp, c_fixed_temp)
                bbox_fake_fixed_val.append(bbox_fake_fixed_temp.clone())
            save_image(torch.cat(bbox_fake_fixed_val), torch.cat(bbox_class_fixed_val), ~torch.cat(padding_mask_fixed_val), training_set.colors,
                        os.path.join(run_dir, f'val_layouts_fake_{cur_nimg//1000:06d}.png'), W_page_fixed_val, H_page_fixed_val)
            save_real_image(torch.cat(bbox_fake_fixed_val), torch.cat(bbox_real_fixed_val), torch.cat(bbox_patch_orig_fixed_val), ~torch.cat(padding_mask_fixed_val),
                            os.path.join(run_dir, f'val_images_fake_{cur_nimg//1000:06d}.png'), W_page_fixed_val, H_page_fixed_val)
            save_real_image_with_background(torch.cat(bbox_fake_fixed_val), torch.cat(bbox_real_fixed_val), torch.cat(bbox_patch_orig_fixed_val), ~torch.cat(padding_mask_fixed_val), torch.cat(background_orig_fixed_val),
                                            os.path.join(run_dir, f'val_images_with_background_fake_{cur_nimg//1000:06d}.png'), W_page_fixed_val, H_page_fixed_val)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                if '_train' in metric:
                    result_dict = metric_main.calc_metric(metric=metric, run_dir=run_dir, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                elif '_val' in metric:
                    result_dict = metric_main.calc_metric(metric=metric, run_dir=run_dir, G=snapshot_data['G_ema'],
                        dataset_kwargs=validation_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
