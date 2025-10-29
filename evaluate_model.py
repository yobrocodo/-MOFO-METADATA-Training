#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Evaluation for MOFO on METADATA Dataset
Generates all metrics, classification reports, and visualizations
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import argparse
import json

from dataset.mutlidomain_baseloader import baseloader
from model.MOFO import MOFO

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def calculate_specificity(y_true, y_pred):
    """Calculate specificity (true negative rate)"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def evaluate_model(model, data_loader, device, args):
    """
    Comprehensive evaluation of the model
    Returns all metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_predictions_proba = []
    all_targets = []
    all_dice_scores = []
    
    print("\nRunning model evaluation on test set...")
    
    with torch.no_grad():
        for IMG, _, MSK, setseq, _ in tqdm(data_loader, desc="Evaluating"):
            IMG = IMG.to(device)
            MSK = MSK.to(device)
            setseq = setseq.to(device)
            
            # Forward pass
            mask_prob_maps, _ = model(IMG)
            mask_prob_maps = torch.sigmoid(mask_prob_maps)
            
            # Get predictions
            for i in range(IMG.shape[0]):
                prob_map = mask_prob_maps[i, setseq[i], :, :].cpu().numpy()
                target_map = MSK[i, setseq[i], :, :].cpu().numpy()
                
                # Binary prediction
                pred_binary = (prob_map > 0.5).astype(np.float32)
                
                # Flatten for classification metrics
                all_predictions_proba.extend(prob_map.flatten())
                all_predictions.extend(pred_binary.flatten())
                all_targets.extend(target_map.flatten())
                
                # Calculate Dice for this sample
                dice = dice_coefficient(target_map, pred_binary)
                all_dice_scores.append(dice)
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_predictions_proba)
    
    print(f"\nTotal pixels evaluated: {len(y_true):,}")
    print(f"Positive pixels (tumor): {np.sum(y_true):,} ({100*np.mean(y_true):.2f}%)")
    
    return y_true, y_pred, y_proba, all_dice_scores

def calculate_all_metrics(y_true, y_pred, y_proba):
    """Calculate all requested metrics"""
    
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Same as sensitivity
    metrics['sensitivity'] = metrics['recall']  # Alias
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    except:
        metrics['roc_auc'] = 0.0
    
    # PR AUC
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    except:
        metrics['pr_auc'] = 0.0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Tumor'],
                yticklabels=['Background', 'Tumor'])
    plt.title('Confusion Matrix\nMOFO Model on METADATA Dataset', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")

def plot_roc_curve(y_true, y_proba, roc_auc, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve\nMOFO Model on METADATA Dataset', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve to {save_path}")

def plot_precision_recall_curve(y_true, y_proba, pr_auc, save_path):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve\nMOFO Model on METADATA Dataset', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PR curve to {save_path}")

def plot_training_curves(log_dir, save_dir):
    """Plot accuracy and loss curves from training logs"""
    import glob
    from tensorboard.backend.event_processing import event_accumulator
    
    # Find tensorboard log files
    log_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not log_files:
        print("⚠ No TensorBoard log files found")
        return
    
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    if 'train/seg_dc' in tags['scalars']:
        train_dc = [(s.step, s.value) for s in ea.Scalars('train/seg_dc')]
        valid_dc = [(s.step, s.value) for s in ea.Scalars('valid/seg_dc')]
        
        epochs_train, loss_train = zip(*train_dc)
        epochs_valid, loss_valid = zip(*valid_dc)
        
        axes[0, 0].plot(epochs_train, loss_train, label='Training', linewidth=2)
        axes[0, 0].plot(epochs_valid, loss_valid, label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Dice Loss', fontsize=11)
        axes[0, 0].set_title('Dice Loss vs Epoch', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Focal loss
    if 'train/seg_fl' in tags['scalars']:
        train_fl = [(s.step, s.value) for s in ea.Scalars('train/seg_fl')]
        valid_fl = [(s.step, s.value) for s in ea.Scalars('valid/seg_fl')]
        
        epochs_train, loss_train = zip(*train_fl)
        epochs_valid, loss_valid = zip(*valid_fl)
        
        axes[0, 1].plot(epochs_train, loss_train, label='Training', linewidth=2)
        axes[0, 1].plot(epochs_valid, loss_valid, label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Focal Loss', fontsize=11)
        axes[0, 1].set_title('Focal Loss vs Epoch', fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Dice score (accuracy proxy)
    if 'valid/dice/METADATA' in tags['scalars']:
        dice_scores = [(s.step, s.value) for s in ea.Scalars('valid/dice/METADATA')]
        epochs, scores = zip(*dice_scores)
        
        axes[1, 0].plot(epochs, scores, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Dice Score', fontsize=11)
        axes[1, 0].set_title('Validation Dice Score vs Epoch', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # Learning rate
    if 'train/lr' in tags['scalars']:
        lr_values = [(s.step, s.value) for s in ea.Scalars('train/lr')]
        epochs, lrs = zip(*lr_values)
        
        axes[1, 1].plot(epochs, lrs, linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 1].set_title('Learning Rate vs Epoch', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.suptitle('MOFO Training Curves on METADATA Dataset', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {save_path}")

def generate_classification_report(y_true, y_pred, save_path):
    """Generate and save detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Background', 'Tumor'],
                                   digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("MOFO Model on METADATA (BUSI) Dataset\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Saved classification report to {save_path}")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num_device', default='0', help='number of device')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='workers number')
    parser.add_argument('--input_size', default=(224, 224), help='input size')
    parser.add_argument('--data_path', default='Multi-Organ Database/', help='dataset path')
    parser.add_argument('--data_configuration', 
                       default='Multi-Organ Database/dataset_config_metadata_only.yaml',
                       help='dataset configuration')
    parser.add_argument('--model_path', 
                       default='output/MOFO_METADATA/saved_model/',
                       help='path to trained model')
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("MOFO Model on METADATA (BUSI) Dataset")
    print("=" * 80)
    
    # Create output directory
    eval_dir = 'evaluation_results'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    _, _, _tt_loader = baseloader(args)
    print(f"Test samples: {len(_tt_loader.dataset)}")
    
    # Load model
    print("\nLoading trained model...")
    model = MOFO(class_num=1, task_prompt='word_embedding')
    
    # Find best model
    import glob
    model_files = glob.glob(os.path.join(args.model_path, '*.pth'))
    if model_files:
        # Get the model with highest dice score from filename
        best_model = max(model_files, key=lambda x: float(x.split('dice_')[-1].replace('.pth', '')))
        print(f"Loading best model: {best_model}")
        model.load_state_dict(torch.load(best_model, map_location=args.device))
    else:
        print("⚠ No trained model found! Using random weights for demonstration.")
    
    model.to(args.device)
    model.eval()
    
    # Evaluate model
    y_true, y_pred, y_proba, dice_scores = evaluate_model(model, _tt_loader, args.device, args)
    
    # Calculate all metrics
    print("\n" + "=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)
    
    metrics = calculate_all_metrics(y_true, y_pred, y_proba)
    
    # Print metrics
    print("\n📊 PERFORMANCE METRICS:")
    print(f"  • Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision:   {metrics['precision']:.4f}")
    print(f"  • Recall:      {metrics['recall']:.4f} (Sensitivity)")
    print(f"  • Specificity: {metrics['specificity']:.4f}")
    print(f"  • F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  • ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"  • PR AUC:      {metrics['pr_auc']:.4f}")
    print(f"  • Mean Dice:   {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    
    # Save metrics to JSON
    metrics['mean_dice'] = float(np.mean(dice_scores))
    metrics['std_dice'] = float(np.std(dice_scores))
    
    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✓ Saved metrics to {eval_dir}/metrics.json")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_confusion_matrix(y_true, y_pred, os.path.join(eval_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_proba, metrics['roc_auc'], os.path.join(eval_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_true, y_proba, metrics['pr_auc'], 
                                os.path.join(eval_dir, 'precision_recall_curve.png'))
    
    # Plot training curves
    log_dir = 'output/MOFO_METADATA/log'
    if os.path.exists(log_dir):
        plot_training_curves(log_dir, eval_dir)
    
    # Generate classification report
    print("\n" + "=" * 80)
    print("GENERATING CLASSIFICATION REPORT")
    print("=" * 80)
    
    report = generate_classification_report(y_true, y_pred, 
                                           os.path.join(eval_dir, 'classification_report.txt'))
    print("\n" + report)
    
    # Create summary document
    summary_path = os.path.join(eval_dir, 'EVALUATION_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MOFO MODEL EVALUATION SUMMARY\n")
        f.write("METADATA (BUSI) Breast Ultrasound Dataset\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write(f"  Test Samples: {len(_tt_loader.dataset)}\n")
        f.write(f"  Total Pixels Evaluated: {len(y_true):,}\n")
        f.write(f"  Positive Pixels (Tumor): {np.sum(y_true):,} ({100*np.mean(y_true):.2f}%)\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:    {metrics['precision']:.4f}\n")
        f.write(f"  Recall:       {metrics['recall']:.4f} (Sensitivity)\n")
        f.write(f"  Specificity:  {metrics['specificity']:.4f}\n")
        f.write(f"  F1 Score:     {metrics['f1_score']:.4f}\n")
        f.write(f"  ROC AUC:      {metrics['roc_auc']:.4f}\n")
        f.write(f"  PR AUC:       {metrics['pr_auc']:.4f}\n")
        f.write(f"  Dice Score:   {metrics['mean_dice']:.4f} ± {metrics['std_dice']:.4f}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write(f"  • metrics.json - All metrics in JSON format\n")
        f.write(f"  • confusion_matrix.png - Confusion matrix visualization\n")
        f.write(f"  • roc_curve.png - ROC curve (TPR vs FPR)\n")
        f.write(f"  • precision_recall_curve.png - Precision-Recall curve\n")
        f.write(f"  • training_curves.png - Training history plots\n")
        f.write(f"  • classification_report.txt - Detailed classification report\n")
        f.write(f"  • EVALUATION_SUMMARY.txt - This summary file\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n✓ Saved evaluation summary to {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {eval_dir}/")
    print("\nFiles created:")
    print(f"  • metrics.json")
    print(f"  • confusion_matrix.png")
    print(f"  • roc_curve.png")
    print(f"  • precision_recall_curve.png")
    print(f"  • training_curves.png")
    print(f"  • classification_report.txt")
    print(f"  • EVALUATION_SUMMARY.txt")

if __name__ == '__main__':
    main()

