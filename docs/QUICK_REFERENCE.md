# Quick Reference Card - ACC Elegant RL Training

## 🚀 Quick Start (< 5 minutes)

```bash
# 1. Extract package
tar -xzf acc_elegant_rl_training.tar.gz
cd acc_elegant_rl_training

# 2. Install dependencies
conda env create -f environment.yml
conda activate acc-elegant-rl

# 3. Run training
python train.py

# 4. Monitor (in another terminal)
tensorboard --logdir=results/
```

---

## 📋 Essential Commands

### Environment Setup
```bash
# Conda
conda env create -f environment.yml
conda activate acc-elegant-rl

# pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# Basic training
python train.py

# Custom episodes
python train.py --training.n_episodes 1000

# Custom config
python train.py --config my_config.yaml

# CPU only
python train.py --training.cpu true
```

### Monitoring
```bash
# TensorBoard dashboard
tensorboard --logdir=results/

# Check logs
tail -f results/ddpg_*/events*

# List models
ls -lh models/
```

---

## ⚙️ Configuration Parameters

### Key Training Parameters
```yaml
training:
  n_episodes: 10              # Episodes to train
  seed: 0                     # Reproducibility
  cpu: false                  # Force CPU

agent:
  alpha: 0.0001               # Actor learning rate
  beta: 0.001                 # Critic learning rate
  batch_size: 128             # Training batch size
  gamma: 0.99                 # Discount factor
  tau: 0.005                  # Soft update
```

### CLI Overrides
```bash
python train.py --training.n_episodes 100 --agent.alpha 0.0002
```

---

## 🐧 Platform-Specific Setup

| Platform | Command |
|----------|---------|
| **Linux** | `apt install python3.10` then follow pip setup |
| **macOS** | `brew install python@3.10` then follow pip setup |
| **Windows** | Download Python 3.10, then use venv |
| **HPC/SLURM** | See DEPLOYMENT_GUIDE.md for sbatch examples |
| **Docker** | `docker build -t acc-rl . && docker run --gpus all acc-rl` |

---

## 🔍 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Import error | `pip install -r requirements.txt --upgrade` |
| GPU not found | `python train.py --training.cpu true` |
| Out of memory | `python train.py --agent.batch_size 64` |
| TensorBoard won't start | Kill process: `pkill tensorboard` |
| Training hangs | `Ctrl+C` to interrupt, check logs |

---

## 📂 Important Files

| File | Purpose |
|------|---------|
| `README.md` | Start here! |
| `SETUP.md` | Installation help |
| `DEPLOYMENT_GUIDE.md` | Production deployment |
| `config.yaml` | Training configuration |
| `train.py` | Main training script |
| `requirements.txt` | Python dependencies |
| `environment.yml` | Conda environment |

---

## 📊 Project Structure

```
acc_elegant_rl_training/
├── train.py              # ← Run this
├── config.yaml           # ← Modify this
├── rl_framework/         # Core library
│   └── Agents/DDPG.py
├── beamline_data/        # Input files
├── results/              # Output (TensorBoard logs)
├── models/               # Output (trained models)
└── logs/                 # Output (CSV logs)
```

---

## ✅ Verification Checklist

```bash
# 1. Check Python
python --version  # Should be 3.10+

# 2. Check imports
python -c "import torch, gymnasium"

# 3. Check GPU (optional)
python -c "import torch; print(torch.cuda.is_available())"

# 4. Test run (1 episode)
python train.py --training.n_episodes 1

# Should complete without errors ✓
```

---

## 🎯 Common Workflows

### Basic Training
```bash
python train.py --training.n_episodes 1000
tensorboard --logdir=results/
```

### Batch Processing
```bash
for seed in {1..5}; do
  python train.py --training.n_episodes 100 --training.seed $seed
done
```

### HPC Submission
```bash
# Edit submit_training.sbatch with your parameters
sbatch submit_training.sbatch
squeue -u $USER  # Check status
```

### Model Testing
```bash
# Load and test trained model
python -c "
import torch
from rl_framework import DDPGAgent
agent = DDPGAgent(state_dim=10, action_dim=1)
# Load weights: agent.load('model.pth')
"
```

---

## 🔗 Documentation Map

```
START HERE → README.md
              ├→ Installation? → SETUP.md
              ├→ Running training? → config.yaml + train.py
              ├→ Deployment? → DEPLOYMENT_GUIDE.md
              └→ Everything listed? → PROJECT_INVENTORY.md
```

---

## 📈 Training Output Structure

After running training, you'll have:

```
results/
└── ddpg_agent_YYYYMMDD_HHMMSS/
    └── events.out.tfevents.***    ← View with TensorBoard

models/
├── ddpg_actor_1000.pth            ← Actor network
├── ddpg_critic_1000.pth           ← Critic network
└── ...

logs/
└── ddpg_eval_*.csv                ← Training metrics
```

View with: `tensorboard --logdir=results/`

---

## 🆘 Getting Help

1. **Quick question?** → Check this Quick Reference
2. **Installation issue?** → See SETUP.md
3. **Configuration help?** → Edit config.yaml or see README.md
4. **Production deployment?** → See DEPLOYMENT_GUIDE.md
5. **Complete file listing?** → See PROJECT_INVENTORY.md
6. **Error codes?** → Check error message in logs/

---

## ⚡ Performance Tips

```bash
# Multi-GPU training (advanced)
export CUDA_VISIBLE_DEVICES=0,1

# Reduce memory usage
python train.py --agent.batch_size 64 --training.cpu true

# Profile performance
python -m cProfile -s cumtime train.py --training.n_episodes 1

# Speed up I/O
export OMP_NUM_THREADS=8
```

---

## 🔐 Security Checklist

- [ ] `.env` file not in repo (for secrets)
- [ ] Config files not world-readable
- [ ] Old models backed up before overwrite
- [ ] SSH keys used for git access
- [ ] No credentials in code or configs

---

## 📞 Version Info

| Item | Value |
|------|-------|
| Package Version | 1.0 |
| Python Target | 3.10+ |
| PyTorch | 2.0+ |
| Status | Production Ready ✅ |
| Last Updated | February 2026 |

---

## 💡 Pro Tips

1. **Always activate environment first**: `conda activate acc-elegant-rl`
2. **Use custom config files**: Create `my_config.yaml` for different runs
3. **Monitor in real-time**: Launch TensorBoard during training
4. **Backup important results**: `tar -czf backup_$(date +%Y%m%d).tar.gz results/ models/`
5. **Use seeds for reproducibility**: `--training.seed 42`
6. **Save configurations**: Document configs used for each run
7. **Version control**: Use git to track changes
8. **Document experiments**: Add notes to results directory

---

**Quick Reference Version**: 1.0  
**Created**: February 2026  
**For complete help**: See README.md, SETUP.md, or DEPLOYMENT_GUIDE.md
