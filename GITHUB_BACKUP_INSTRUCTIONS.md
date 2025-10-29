# GitHub Backup Instructions

## Current Status
Your repository is currently linked to the original MOFO repository:
- `origin: https://github.com/Git-HB-CHEN/MOFO.git`

To back up YOUR work, you need to create your own GitHub repository and push to it.

## Step-by-Step Instructions

### 1. Create a New GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** button in the top right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `MOFO-METADATA-Training` (or any name you prefer)
   - **Description**: "MOFO model training on METADATA (BUSI) dataset"
   - **Visibility**: Choose **Private** (recommended) or **Public**
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### 2. Update Your Local Repository

After creating the repository, GitHub will show you commands. Use these:

#### Option A: If you want to keep the original MOFO as a reference
```bash
# Add your new repository as a new remote called "backup"
git remote add backup https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Stage all your changes
git add .

# Commit your work
git commit -m "Set up METADATA dataset for training

- Extracted and organized 589 BUSI samples
- Created training script (train_metadata.py)
- Updated configurations
- Fixed JSON files
- Added comprehensive documentation"

# Push to your backup repository
git push -u backup main
```

#### Option B: If you want to replace the origin (recommended)
```bash
# Remove the old origin
git remote remove origin

# Add your repository as the new origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Stage all your changes
git add .

# Commit your work
git commit -m "Set up METADATA dataset for training

- Extracted and organized 589 BUSI samples  
- Created training script (train_metadata.py)
- Updated configurations
- Fixed JSON files
- Added comprehensive documentation"

# Push to your repository
git push -u origin main
```

### 3. Files That Will Be Backed Up

✅ **Included in backup**:
- All Python scripts
- Configuration files (YAML, JSON)
- Documentation (README, SETUP_SUMMARY)
- Requirements.txt
- Model architecture code

❌ **Excluded from backup** (too large or not needed):
- METADATA.zip (389 MB)
- Extracted dataset files
- Model weight files (.pth)
- Virtual environment (mofo-env/)
- Training outputs (output/)
- Temporary files

### 4. Quick Commands (Copy-Paste Ready)

**Replace these placeholders:**
- `YOUR_USERNAME` → Your GitHub username
- `YOUR_REPO_NAME` → Your repository name

```bash
# Change to your project directory (if not already there)
cd "C:\Users\yoavb\OneDrive\Documents\University\Final Project\unet\MOFO"

# Configure git if not already done
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Update remote
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Add and commit all changes
git add .
git commit -m "Set up METADATA dataset for MOFO training"

# Push to GitHub
git push -u origin main
```

### 5. Verify Backup

After pushing, verify on GitHub:
1. Go to your repository URL
2. Check that all files are there
3. Verify the commit message is correct

## Important Notes

### Large Files
The `.gitignore` file is configured to exclude:
- Dataset files (METADATA.zip, extracted images)
- Model weights (.pth files)
- Virtual environment
- Training outputs

**Why?** GitHub has a 100MB file size limit. These files should be stored separately (e.g., Google Drive, OneDrive).

### Keeping Dataset Backed Up Separately

Your METADATA dataset is in:
```
Multi-Organ Database/Dataset_METADATA/
```

Consider backing this up to:
- **OneDrive**: You're already using OneDrive for this project
- **Google Drive**: Good for sharing with collaborators
- **External hard drive**: Local backup

### Future Updates

When you make changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

## Troubleshooting

### "Permission denied" error
You may need to authenticate with GitHub:
- Use [Personal Access Token](https://github.com/settings/tokens) instead of password
- Or set up [SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

### "Repository already exists" error
The repository name is taken. Choose a different name.

### "Large files" warning
If you see warnings about large files, they're already in `.gitignore`. 
Use `git rm --cached filename` to remove them from tracking.

---

**Need Help?** 
- GitHub Docs: https://docs.github.com/
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf

