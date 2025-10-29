#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for METADATA (BUSI) dataset
Modified from original train.py for single dataset training
"""

import os
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset.mutlidomain_baseloader import baseloader
from model.MOFO import MOFO
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import SelectedDSCLoss, SelectedFLoss
from utils.metric import metric_pixel_dice

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device')
parser.add_argument('--num_device', default='0', help='number of device')
parser.add_argument('--log_flag', default=True, help='flag of log')
parser.add_argument('--save_flag', default=True, help='flag of save model')
parser.add_argument('--log_name', default='MOFO_METADATA', help='name of log')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument("--epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=100, type=int, help='Number of training epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--warmup_epoch', default=5, type=int, help='number of warmup epochs')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')
parser.add_argument('--input_size', default=(224, 224), help='input size')
parser.add_argument('--data_path', default='Multi-Organ Database/',
                    help='path of the dataset')
parser.add_argument('--data_configuration', default='Multi-Organ Database/dataset_config_metadata_only.yaml',
                    help='configuration of the dataset')
args = parser.parse_args()

# Setup device
if torch.cuda.is_available():
    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_device
    args.device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    args.device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Setup logging
if args.log_flag:
    os.makedirs(os.path.join('output/', args.log_name, 'log/'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join('output/', args.log_name, 'log/'))
    print(f'Writing Tensorboard logs to output/{args.log_name}/log/')

# Load data
print("\nLoading METADATA dataset...")
_tn_loader, _vd_loader, _tt_loader = baseloader(args)
print(f"Dataset loaded:")
print(f"  Training: {len(_tn_loader.dataset)} samples")
print(f"  Validation: {len(_vd_loader.dataset)} samples")
print(f"  Test: {len(_tt_loader.dataset)} samples")
print(f"  Number of classes: {args.domian_num}")

# Create model - only 1 class for METADATA
model = MOFO(class_num=args.domian_num, task_prompt='word_embedding')

# Try to load pretrained weights
try:
    model.load_from(pretrained_path='model/cswin_small_224.pth')
    print("[OK] Loaded pretrained CSwin backbone weights")
except Exception as e:
    print("[WARNING] Could not load pretrained weights")
    print("  Continuing with random initialization...")

# Initialize organ embedding (must be 512 dimensions for word_embedding mode)
organ_embedding = torch.randn(args.domian_num, 512).to(args.device)
model.organ_embedding.data = organ_embedding.float()
model.to(args.device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Setup loss functions
loss_seg_DC = SelectedDSCLoss().to(args.device)
loss_seg_FL = SelectedFLoss().to(args.device)
loss_cls_CE = nn.CrossEntropyLoss().to(args.device)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

best_score = 0

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

while args.epoch < args.max_epoch:
    # Training phase
    scheduler.step()
    model.train()
    loss_seg_dc_ave_tn = 0
    loss_seg_fl_ave_tn = 0
    loss_cls_ce_ave_tn = 0

    epoch_tn_iterator = tqdm(_tn_loader, desc=f"Epoch {args.epoch+1}/{args.max_epoch} [Train]")
    for step, (IMG, MSK1ch, MSK, setseq, _) in enumerate(epoch_tn_iterator):
        IMG, MSK1ch, MSK, setseq = IMG.to(args.device), MSK1ch.to(args.device), MSK.to(args.device), setseq.to(args.device)

        mask_prob_maps, classification_prob_maps = model(IMG)
        
        term_seg_DC = loss_seg_DC.forward(mask_prob_maps, MSK, setseq)
        term_seg_FL = loss_seg_FL.forward(mask_prob_maps, MSK, setseq)
        term_cls_CE = loss_cls_CE.forward(classification_prob_maps, setseq)

        loss = term_seg_DC + term_seg_FL * 0.5 + term_cls_CE * 0.3
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_seg_dc_ave_tn += term_seg_DC.item()
        loss_seg_fl_ave_tn += term_seg_FL.item()
        loss_cls_ce_ave_tn += term_cls_CE.item()
        
        epoch_tn_iterator.set_postfix({
            'DC': f'{term_seg_DC.item():.4f}',
            'FL': f'{term_seg_FL.item():.4f}',
            'CE': f'{term_cls_CE.item():.4f}'
        })

    loss_seg_dc_tn = loss_seg_dc_ave_tn / len(epoch_tn_iterator)
    loss_seg_fl_tn = loss_seg_fl_ave_tn / len(epoch_tn_iterator)
    loss_cls_ce_tn = loss_cls_ce_ave_tn / len(epoch_tn_iterator)

    print(f'Train Epoch {args.epoch+1}: DC={loss_seg_dc_tn:.5f}, FL={loss_seg_fl_tn:.5f}, CE={loss_cls_ce_tn:.5f}')

    # Validation phase
    model.eval()
    loss_seg_dc_ave_vd = 0
    loss_seg_fl_ave_vd = 0
    loss_cls_ce_ave_vd = 0
    dsc_list_vd = []
    dsc_dict_vd = dict()

    epoch_vd_iterator = tqdm(_vd_loader, desc=f"Epoch {args.epoch+1}/{args.max_epoch} [Valid]")
    with torch.no_grad():
        for step, (IMG, _, MSK, setseq, uslabel) in enumerate(epoch_vd_iterator):
            IMG, MSK, setseq = IMG.to(args.device), MSK.to(args.device), setseq.to(args.device)

            mask_prob_maps, classification_prob_maps = model(IMG)
            term_seg_DC = loss_seg_DC.forward(mask_prob_maps, MSK, setseq)
            term_seg_FL = loss_seg_FL.forward(mask_prob_maps, MSK, setseq)
            term_cls_CE = loss_cls_CE.forward(classification_prob_maps, setseq)

            loss_seg_dc_ave_vd += term_seg_DC.item()
            loss_seg_fl_ave_vd += term_seg_FL.item()
            loss_cls_ce_ave_vd += term_cls_CE.item()

            _loss_list_vd = metric_pixel_dice(mask_prob_maps, MSK, setseq)
            for k, v in zip(uslabel, _loss_list_vd):
                if k not in dsc_dict_vd.keys():
                    dsc_dict_vd[k] = []
                dsc_dict_vd[k].append(v)

    loss_seg_dc_vd = loss_seg_dc_ave_vd / len(epoch_vd_iterator)
    loss_seg_fl_vd = loss_seg_fl_ave_vd / len(epoch_vd_iterator)
    loss_cls_ce_vd = loss_cls_ce_ave_vd / len(epoch_vd_iterator)

    _valid_average_dsc = []
    print(f'Valid Epoch {args.epoch+1}: DC={loss_seg_dc_vd:.5f}, FL={loss_seg_fl_vd:.5f}, CE={loss_cls_ce_vd:.5f}')
    for k, v in dsc_dict_vd.items():
        print(f'  {k} - Dice: {np.mean(v):.5f} ± {np.std(v):.5f}')
        _valid_average_dsc.append(np.mean(v))

    # Save results
    os.makedirs(os.path.join('output/', args.log_name), exist_ok=True)
    with open(os.path.join('output/', args.log_name, '_result_valid.txt'), 'a') as f:
        f.write(f"Epoch {args.epoch}: ")
        for k, v in dsc_dict_vd.items():
            f.write(f'{k}-{np.mean(v):.5f}-{np.std(v):.5f} ')
        f.write('\n')

    # TensorBoard logging
    if args.log_flag:
        writer.add_scalar('train/seg_dc', loss_seg_dc_tn, args.epoch)
        writer.add_scalar('train/seg_fl', loss_seg_fl_tn, args.epoch)
        writer.add_scalar('train/cls_ce', loss_cls_ce_tn, args.epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], args.epoch)

        writer.add_scalar('valid/seg_dc', loss_seg_dc_vd, args.epoch)
        writer.add_scalar('valid/seg_fl', loss_seg_fl_vd, args.epoch)
        writer.add_scalar('valid/cls_ce', loss_cls_ce_vd, args.epoch)
        for k, v in dsc_dict_vd.items():
            writer.add_scalar(f'valid/dice/{k}', np.mean(v), args.epoch)

    # Save best model
    if args.save_flag and args.epoch >= args.warmup_epoch and np.mean(_valid_average_dsc) >= best_score:
        best_score = np.mean(_valid_average_dsc)
        os.makedirs(os.path.join('output/', args.log_name, 'saved_model/'), exist_ok=True)
        save_path = os.path.join('output/', args.log_name, 'saved_model/', f'model_epoch_{args.epoch}_dice_{best_score:.4f}.pth')
        torch.save(model.state_dict(), save_path)
        print(f'[SAVED] Best model: {save_path} (Dice: {best_score:.4f})')

    args.epoch += 1

print("\n" + "=" * 60)
print("Training completed!")
print(f"Best validation Dice score: {best_score:.4f}")
print("=" * 60)

