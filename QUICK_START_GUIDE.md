# 🚀 Quick Start Guide - MOFO METADATA Training

## ✅ Everything is Ready!

Your MOFO model is set up and ready to train on the METADATA (BUSI) dataset with **589 breast ultrasound images**.

---

## 🎯 Step 1: Start Training NOW

```bash
python run_with_cuda_patch.py train_metadata.py
```

That's it! Training will start immediately.

---

## 📊 What to Expect

### Training Progress
- **Epochs**: 100 (will take several hours)
- **Batch size**: 8 images per batch
- **Learning rate**: 0.0001 with warmup

### Console Output
You'll see:
```
Epoch 1/100 [Train]: 100%|████| DC=0.3245, FL=0.4532, CE=0.1234
Train Epoch 1: DC=0.3245, FL=0.4532, CE=0.1234
Epoch 1/100 [Valid]: 100%|████|
Valid Epoch 1: DC=0.3421, FL=0.4312, CE=0.1123
  METADATA - Dice: 0.6578 ± 0.1234
✓ Saved best model: output/MOFO_METADATA/saved_model/model_epoch_1_dice_0.6578.pth
```

### Output Files
- **Best models**: `output/MOFO_METADATA/saved_model/`
- **TensorBoard logs**: `output/MOFO_METADATA/log/`
- **Results file**: `output/MOFO_METADATA/_result_valid.txt`

---

## 📈 Step 2: Monitor Training (Optional)

### Open TensorBoard
In a **new terminal**:
```bash
tensorboard --logdir output/MOFO_METADATA/log/
```
Then open: http://localhost:6006

---

## 💾 Step 3: Backup to GitHub

### Quick Setup
1. **Create a new repository** on GitHub:
   - Go to https://github.com/new
   - Name it: `MOFO-METADATA-Training`
   - Make it **Private**
   - Click "Create repository"

2. **Push your code** (replace YOUR_USERNAME with your GitHub username):
   ```bash
   git remote remove origin
   git remote add origin https://github.com/YOUR_USERNAME/MOFO-METADATA-Training.git
   git commit -m "Set up METADATA dataset for MOFO training"
   git push -u origin main
   ```

📖 **Detailed instructions**: See `GITHUB_BACKUP_INSTRUCTIONS.md`

---

## 🎓 Your Dataset

- **Total samples**: 589
- **Training**: 412 samples (70%)
- **Validation**: 88 samples (15%)
- **Test**: 89 samples (15%)
- **Task**: Breast tumor segmentation
- **Source**: BUSI (Breast Ultrasound Images) dataset

Location: `Multi-Organ Database/Dataset_METADATA/`

---

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `SETUP_SUMMARY.md` | Complete setup details and troubleshooting |
| `GITHUB_BACKUP_INSTRUCTIONS.md` | Step-by-step GitHub backup guide |
| `train_metadata.py` | Your training script |
| `test_metadata_setup.py` | Verification script |

---

## ⚙️ Training Options

### Change batch size (if out of memory):
```bash
python run_with_cuda_patch.py train_metadata.py --batch_size 4
```

### Change learning rate:
```bash
python run_with_cuda_patch.py train_metadata.py --lr 0.001
```

### Change number of epochs:
```bash
python run_with_cuda_patch.py train_metadata.py --max_epoch 200
```

### Combine options:
```bash
python run_with_cuda_patch.py train_metadata.py --batch_size 16 --lr 0.0005 --max_epoch 150
```

---

## 🔍 Verify Everything Works

Run the test script:
```bash
python run_with_cuda_patch.py test_metadata_setup.py
```

Expected output:
```
✓ ALL TESTS PASSED - Ready for training!
```

---

## 🆘 Troubleshooting

### Out of Memory Error
→ Reduce batch size: `--batch_size 4`

### Training Too Slow
→ This is normal on CPU. Each epoch takes ~2-5 minutes.

### Model Not Improving
→ Training needs more epochs (50-100) to converge

### Need Help?
→ Check `SETUP_SUMMARY.md` for detailed troubleshooting

---

## 📊 Expected Results

After training completes (~100 epochs):
- **Validation Dice Score**: 0.70 - 0.85
- **Best model saved**: `output/MOFO_METADATA/saved_model/model_epoch_XX_dice_X.XXXX.pth`

---

## ✨ Summary

✅ Dataset prepared (589 samples)  
✅ Training script ready  
✅ Verification passed  
✅ Documentation complete  
✅ Git staged for backup  

**You're all set! Start training now:** 
```bash
python run_with_cuda_patch.py train_metadata.py
```

Good luck with your training! 🎉

