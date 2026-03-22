# Deployment Guide - ACC Elegant RL Training Package

Complete guide for deploying the ACC Elegant RL Training package to production environments.

## 🎯 Deployment Overview

This guide covers:
- Transferring the package to target systems
- Configuring for specific environments
- Running in production
- Monitoring and troubleshooting
- Scaling to multiple experiments

---

## 📦 Package Preparation

### 1. Create Distribution Archive

```bash
# From the parent directory
cd /path/to/parent
tar -czf acc_elegant_rl_training.tar.gz acc_elegant_rl_training/

# Or create zip
zip -r acc_elegant_rl_training.zip acc_elegant_rl_training/
```

### 2. Verify Contents

```bash
# Check archive integrity
tar -tzf acc_elegant_rl_training.tar.gz | head -20

# Or
unzip -l acc_elegant_rl_training.zip | head -20
```

### 3. Add .gitignore (Optional)

Create `.gitignore` in the deployment package:

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Outputs
results/*
models/*
logs/*
*.csv
*.log
*.pkl

# Temporary
.DS_Store
*.tmp
.ipynb_checkpoints/

# Except for structure
!results/.gitkeep
!models/.gitkeep
!logs/.gitkeep
```

---

## 🚀 Deployment to New Machine

### Step 1: Transfer Package

**Option A: Via SSH/SCP**
```bash
# From local machine
scp -r acc_elegant_rl_training.tar.gz user@remote:/path/to/deployment/

# On remote machine
tar -xzf acc_elegant_rl_training.tar.gz
cd acc_elegant_rl_training
```

**Option B: Via Git**
```bash
# If using version control
git clone https://github.com/username/acc-elegant-rl.git
cd acc_elegant_rl_training
```

**Option C: Manual Download**
```bash
# Download from cloud storage (AWS S3, Google Drive, etc.)
wget https://example.com/acc_elegant_rl_training.tar.gz
tar -xzf acc_elegant_rl_training.tar.gz
cd acc_elegant_rl_training
```

### Step 2: Verify Transfer

```bash
# Check all files present
ls -la
ls rl_framework/
ls *.lte *.ele

# Verify key files
test -f train.py && echo "✓ train.py present"
test -f config_manager.py && echo "✓ config_manager.py present"
test -f machine.lte && echo "✓ machine.lte present"
test -f requirements.txt && echo "✓ requirements.txt present"
```

### Step 3: Setup Environment

Follow **SETUP.md** for platform-specific instructions:

```bash
# Quick setup (Linux/macOS)
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Run verification script
python -c "
import torch
import gymnasium
from rl_framework import ACCElegantEnvironment, DDPGAgent
print('✅ Deployment verification successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## ⚙️ Environment Configuration

### 1. Default Configuration

The package comes with `config.yaml`:

```bash
cat config.yaml | head -30
```

### 2. Customize Configuration

Create `local_config.yaml`:

```yaml
training:
  n_episodes: 1000
  seed: 42
  cpu: false

agent:
  alpha: 0.0001
  beta: 0.001
  batch_size: 128
  gamma: 0.99
  tau: 0.005

environment:
  init_num_particles: 1000
```

### 3. Run with Custom Config

```bash
python train.py --config local_config.yaml
```

---

## 🏃 Running Training

### Single Run

```bash
# Basic run with defaults
python train.py

# With custom episodes
python train.py --training.n_episodes 1000

# With seed for reproducibility
python train.py --training.n_episodes 1000 --training.seed 42
```

### Background/Detached Mode

**Linux/macOS:**
```bash
# Run in background
nohup python train.py > training.log 2>&1 &

# Check progress
tail -f training.log

# Get job ID
ps aux | grep train.py
```

**macOS with Screen:**
```bash
screen -S training
python train.py
# Detach: Ctrl+A then D
# Reattach: screen -r training
```

### Multiple Runs (Hyperparameter Search)

Create `run_experiments.sh`:

```bash
#!/bin/bash

# Hyperparameter search
for lr in 0.0001 0.0002 0.0005; do
  for batch_size in 64 128 256; do
    echo "Running with lr=$lr, batch_size=$batch_size"
    python train.py \
      --training.n_episodes 1000 \
      --training.seed $RANDOM \
      --agent.alpha $lr \
      --agent.batch_size $batch_size
  done
done
```

Run with: `bash run_experiments.sh`

---

## 📊 Monitoring Training

### 1. Real-Time Dashboard (TensorBoard)

```bash
# In one terminal - run training
python train.py --training.n_episodes 1000

# In another terminal - launch TensorBoard
tensorboard --logdir=results/ --port=6006

# Access: http://localhost:6006
```

### 2. Log Files

```bash
# Check training logs
tail -f results/ddpg_*/events.out.tfevents.*

# Check model logs
tail -f logs/ddpg_eval_*.csv
```

### 3. Output Directory

```bash
# Training artifacts
tree results/
tree models/
tree logs/

# Count episodes
ls models/ | grep actor | wc -l
```

### 4. Custom Monitoring Script

Create `monitor_training.py`:

```python
#!/usr/bin/env python
import os
import glob
from datetime import datetime

def monitor():
    print(f"\n{'='*60}")
    print(f"Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check results
    result_dirs = glob.glob("results/*/")
    if result_dirs:
        print(f"\n📊 Results: {len(result_dirs)} runs")
        for d in sorted(result_dirs)[-3:]:  # Last 3 runs
            print(f"  - {os.path.basename(d.rstrip('/'))}")
    
    # Check models
    models = glob.glob("models/ddpg_actor_*.pth")
    if models:
        print(f"\n🤖 Models: {len(models)} checkpoints")
        latest = max(models, key=os.getctime)
        print(f"  Latest: {os.path.basename(latest)}")
        size_mb = os.path.getsize(latest) / (1024*1024)
        print(f"  Size: {size_mb:.1f} MB")
    
    # Check logs
    logs = glob.glob("logs/ddpg_eval_*.csv")
    if logs:
        print(f"\n📈 Logs: {len(logs)} log files")
        with open(logs[-1], 'r') as f:
            lines = f.readlines()
            print(f"  Latest entries: {len(lines)-1} episodes")

if __name__ == "__main__":
    monitor()
```

Run with: `python monitor_training.py`

---

## 🖥️ HPC Cluster Deployment

### SLURM Submission

Create `submit_training.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=acc_elegant_rl
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu

# Load modules
module load python/3.10
module load cuda/11.8

# Setup environment
source venv/bin/activate

# Run training
python train.py \
  --training.n_episodes 10000 \
  --training.seed $SLURM_ARRAY_TASK_ID

echo "Training completed successfully"
```

Submit with:
```bash
sbatch submit_training.sbatch

# Array job (multiple runs)
sbatch --array=1-10 submit_training.sbatch

# Check status
squeue -u $USER

# Cancel job
scancel <JOB_ID>
```

### PBS Submission

Create `submit_training.pbs`:

```bash
#!/bin/bash
#PBS -N acc_elegant_rl
#PBS -l select=1:ngpus=1:cuda_compute_capability=7.0
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o training_$PBS_JOBID.log

cd $PBS_O_WORKDIR

# Load modules
module load python/3.10
module load cuda/11.8

# Setup environment
source venv/bin/activate

# Run training
python train.py \
  --training.n_episodes 10000 \
  --training.seed $PBS_ARRAY_INDEX
```

Submit with:
```bash
qsub submit_training.pbs

# Array job
qsub -J 1-10 submit_training.pbs

# Check status
qstat

# Cancel job
qdel <JOB_ID>
```

---

## 📈 Results Management

### 1. Organize Results

```bash
# Create dated directory
mkdir results/run_$(date +%Y%m%d_%H%M%S)

# Move training outputs
mv results/ddpg_*/ results/run_$(date +%Y%m%d_%H%M%S)/
```

### 2. Backup Results

```bash
# Tar results
tar -czf backup_$(date +%Y%m%d).tar.gz results/ models/ logs/

# Upload to cloud
aws s3 cp backup_*.tar.gz s3://my-bucket/backups/
```

### 3. Compare Runs

Create `compare_runs.py`:

```python
import pandas as pd
import glob

logs = sorted(glob.glob("logs/ddpg_eval_*.csv"))

for log in logs[-3:]:  # Last 3 runs
    df = pd.read_csv(log)
    print(f"\n{log}")
    print(df[['episode', 'reward', 'loss']].tail(5))
```

---

## 🔧 Troubleshooting Deployment

### Issue: Package Won't Extract

```bash
# Check integrity
tar -tzf acc_elegant_rl_training.tar.gz > /dev/null

# Try with verbose
tar -xzvf acc_elegant_rl_training.tar.gz
```

### Issue: Import Errors After Transfer

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue: GPU Not Available

```bash
# Verify CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
python train.py --training.cpu true
```

### Issue: Out of Disk Space

```bash
# Check available space
df -h

# Clean old results
rm -rf results/old_run_*/
rm -rf models/old_*.pth

# Archive and compress
tar -czf archive_$(date +%Y%m%d).tar.gz results/ models/
rm -rf results/ models/
```

### Issue: Training Hangs

```bash
# Check process
ps aux | grep train.py

# Kill process if needed
kill -9 <PID>

# Check logs for errors
tail -100 results/ddpg_*/events*
```

---

## 🔒 Security Considerations

### 1. File Permissions

```bash
# Restrict access
chmod 755 .
chmod 755 *.py
chmod 700 config.yaml  # If contains secrets

# Restrict group/others
chmod go-rwx config.yaml
```

### 2. Data Privacy

```bash
# Secure deletion of sensitive data
shred -vfz -n 10 old_model.pth

# Or use macOS Secure Empty Trash
rm models/old_*.pth  # macOS: uses secure deletion by default
```

### 3. Network Security

For remote deployment:
```bash
# Use SSH keys instead of passwords
ssh-copy-id -i ~/.ssh/id_rsa.pub user@remote

# Disable password login in sshd_config
# PasswordAuthentication no
```

---

## 📊 Performance Optimization

### 1. GPU Optimization

```bash
# Set GPU memory growth to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

# Use mixed precision
python train.py --agent.use_amp true
```

### 2. Multi-GPU Training

```bash
# Use distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Modify train.py to use DataParallel or DistributedDataParallel
# (requires code changes)
```

### 3. Profiling

```python
# Add to train.py for profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... training code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## ✅ Deployment Checklist

Before production deployment:

- [ ] Package created and verified (`tar.gz` or `zip`)
- [ ] All files present in archive (train.py, config.yaml, machine.lte, track.ele)
- [ ] SETUP.md reviewed for target platform
- [ ] Python 3.10+ available on target system
- [ ] Requirements.txt installable without errors
- [ ] Configuration file created (local_config.yaml)
- [ ] Verification test runs successfully (1 episode)
- [ ] TensorBoard launches without errors
- [ ] GPU detected (if GPU deployment)
- [ ] Sufficient disk space available (10+ GB for results)
- [ ] Monitoring/logging strategy defined
- [ ] Backup strategy in place
- [ ] Rollback plan documented

---

## 🎓 Quick Reference

### Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Run training
python train.py

# Monitor with TensorBoard
tensorboard --logdir=results/

# Check GPU
nvidia-smi

# View latest logs
tail -f results/ddpg_*/events*

# List models
ls -lh models/

# Backup results
tar -czf backup_$(date +%Y%m%d).tar.gz results/ models/
```

---

## 📚 Related Documentation

- **README.md**: Quick start guide
- **SETUP.md**: Platform-specific installation
- **config.yaml**: Configuration reference
- **train.py**: Source code documentation

---

**Deployment Guide Version**: 1.0
**Last Updated**: February 2026
**Status**: Production Ready ✅

For issues or questions, refer to SETUP.md or README.md.
