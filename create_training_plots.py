#!/usr/bin/env python3
"""
Create clear, standalone training plots from validation results
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Read validation results
results_file = 'output/MOFO_METADATA/_result_valid.txt'

epochs = []
dice_scores = []
std_scores = []

print("Reading training results...")
with open(results_file, 'r') as f:
    for line in f:
        if line.strip():
            # Parse line: "Epoch 1: METADATA-0.12345-0.67890"
            parts = line.split(':')
            epoch = int(parts[0].split()[1])
            metrics = parts[1].strip().split('-')
            dice = float(metrics[1])
            std = float(metrics[2])
            
            epochs.append(epoch)
            dice_scores.append(dice)
            std_scores.append(std)

print(f"Found {len(epochs)} epochs")

# Create output directory
os.makedirs('evaluation_results', exist_ok=True)

# 1. Dice Score vs Epoch (This is the accuracy metric for segmentation)
plt.figure(figsize=(12, 7))
plt.plot(epochs, dice_scores, linewidth=2.5, marker='o', markersize=4, 
         color='#2E86DE', label='Validation Dice Score')
plt.fill_between(epochs, 
                  np.array(dice_scores) - np.array(std_scores),
                  np.array(dice_scores) + np.array(std_scores),
                  alpha=0.2, color='#2E86DE', label='±1 Std Dev')
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Dice Score (Segmentation Accuracy)', fontsize=14, fontweight='bold')
plt.title('Model Accuracy vs Training Epoch\nMOFO on METADATA Dataset', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0, 1])
plt.xlim([0, max(epochs)])

# Add annotations for best score
best_idx = np.argmax(dice_scores)
best_epoch = epochs[best_idx]
best_dice = dice_scores[best_idx]
plt.annotate(f'Best: {best_dice:.4f}\nEpoch {best_epoch}',
             xy=(best_epoch, best_dice),
             xytext=(best_epoch-5, best_dice-0.1),
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

plt.tight_layout()
plt.savefig('evaluation_results/accuracy_vs_epoch.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation_results/accuracy_vs_epoch.png")
plt.close()

# 2. Loss vs Epoch (Inverse of Dice - lower is better)
loss_values = [1 - d for d in dice_scores]
plt.figure(figsize=(12, 7))
plt.plot(epochs, loss_values, linewidth=2.5, marker='o', markersize=4,
         color='#EE5A6F', label='Validation Loss (1 - Dice)')
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
plt.title('Model Loss vs Training Epoch\nMOFO on METADATA Dataset', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0, 1])
plt.xlim([0, max(epochs)])

# Add annotations for best (lowest) loss
best_loss_idx = np.argmin(loss_values)
best_loss_epoch = epochs[best_loss_idx]
best_loss = loss_values[best_loss_idx]
plt.annotate(f'Best: {best_loss:.4f}\nEpoch {best_loss_epoch}',
             xy=(best_loss_epoch, best_loss),
             xytext=(best_loss_epoch+5, best_loss+0.1),
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

plt.tight_layout()
plt.savefig('evaluation_results/loss_vs_epoch.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation_results/loss_vs_epoch.png")
plt.close()

# 3. Combined Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Accuracy subplot
ax1.plot(epochs, dice_scores, linewidth=2.5, marker='o', markersize=4, 
         color='#2E86DE')
ax1.fill_between(epochs, 
                  np.array(dice_scores) - np.array(std_scores),
                  np.array(dice_scores) + np.array(std_scores),
                  alpha=0.2, color='#2E86DE')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Dice Score (Accuracy)', fontsize=13, fontweight='bold')
ax1.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1])
ax1.axhline(y=best_dice, color='g', linestyle='--', alpha=0.5, 
            label=f'Best: {best_dice:.4f}')
ax1.legend(fontsize=10)

# Loss subplot
ax2.plot(epochs, loss_values, linewidth=2.5, marker='o', markersize=4,
         color='#EE5A6F')
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Loss (1 - Dice)', fontsize=13, fontweight='bold')
ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1])
ax2.axhline(y=best_loss, color='g', linestyle='--', alpha=0.5,
            label=f'Best: {best_loss:.4f}')
ax2.legend(fontsize=10)

fig.suptitle('MOFO Training Progress on METADATA Dataset', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('evaluation_results/training_progress.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation_results/training_progress.png")
plt.close()

print("\n" + "="*60)
print("TRAINING PLOTS CREATED")
print("="*60)
print(f"\nTotal Epochs: {len(epochs)}")
print(f"Best Dice Score: {best_dice:.4f} (Epoch {best_epoch})")
print(f"Final Dice Score: {dice_scores[-1]:.4f} (Epoch {epochs[-1]})")
print(f"\nFiles created in evaluation_results/:")
print("  • accuracy_vs_epoch.png - Dice score (accuracy) over training")
print("  • loss_vs_epoch.png - Loss over training")
print("  • training_progress.png - Combined accuracy and loss")

