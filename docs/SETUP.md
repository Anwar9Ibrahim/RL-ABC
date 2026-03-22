# Setup Instructions - ACC Elegant RL Training

Complete platform-specific setup guide for `acc_elegant_rl_training` deployment package.

## 📋 Prerequisites

All platforms require:
- Python 3.10 or higher
- pip or conda package manager
- 2GB free disk space
- Internet connection for initial setup

## 🐧 Linux Setup

### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip
sudo apt-get install -y build-essential git
```

**CentOS/RHEL:**
```bash
sudo yum install -y python310 python310-devel
sudo yum groupinstall -y "Development Tools"
```

### 2. Python Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('✓ PyTorch installed')"
```

### 4. GPU Support (Optional - NVIDIA)

```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. Run Training

```bash
python train.py --training.n_episodes 100
```

---

## 🍎 macOS Setup

### 1. System Dependencies

**Install Homebrew:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install Python:**
```bash
brew install python@3.10
brew install git
```

### 2. Python Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# For macOS with Apple Silicon (M1/M2)
pip install --upgrade tensorflow-macos tensorflow-metal

# Verify installation
python -c "import torch; print('✓ PyTorch installed')"
```

### 4. GPU Support (Apple Metal)

```bash
# If using Apple Silicon, PyTorch will use Metal automatically
python -c "import torch; print(torch.backends.mps.is_available())"

# Force CPU if needed
python train.py --training.cpu true
```

### 5. Run Training

```bash
python train.py --training.n_episodes 100
```

---

## 🪟 Windows Setup

### Option A: Native Python (Recommended)

#### 1. Install Python

1. Download Python 3.10 from [python.org](https://www.python.org/downloads/)
2. Run installer, **check "Add Python to PATH"**
3. Verify: Open Command Prompt and run `python --version`

#### 2. Virtual Environment

```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

#### 3. Install Dependencies

```cmd
# Install requirements
pip install -r requirements.txt

# Verify
python -c "import torch; print('✓ PyTorch installed')"
```

#### 4. Run Training

```cmd
python train.py --training.n_episodes 100
```

### Option B: Windows Subsystem for Linux (WSL2)

Better for GPU support and Linux compatibility.

#### 1. Enable WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install
wsl --set-default-version 2
```

#### 2. Install Linux Distribution

```powershell
wsl --install -d Ubuntu-22.04
```

#### 3. Setup in WSL2

```bash
# Inside WSL terminal
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y build-essential git

# Then follow Linux setup above
```

---

## 🐳 Docker Setup (All Platforms)

### Create Dockerfile

Create file named `Dockerfile` in project root:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train.py"]
```

### Build and Run

```bash
# Build image
docker build -t acc-elegant-rl:latest .

# Run training
docker run --gpus all acc-elegant-rl:latest --training.n_episodes 100

# Interactive mode
docker run -it --gpus all acc-elegant-rl:latest /bin/bash
```

---

## 🎯 Conda Setup (All Platforms)

### 1. Install Conda

Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

### 2. Create Environment

```bash
# Using conda environment file
conda env create -f environment.yml

# Activate
conda activate acc-elegant-rl
```

### 3. Verify

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Run Training

```bash
python train.py
```

---

## ✅ Verification Steps

### 1. Basic Import Test

```bash
python -c "
import sys
import torch
import gymnasium
import numpy as np

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Gymnasium: {gymnasium.__version__}')
print(f'NumPy: {np.__version__}')
print('✓ All imports successful')
"
```

### 2. GPU Check

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.get_device_name(0)}')
"
```

### 3. File Check

```bash
# Linux/macOS
ls -lh *.lte *.ele
ls -lh *.py
ls -lh config.*

# Windows
dir *.lte *.ele
dir *.py
dir config.*
```

### 4. Quick Training Test

```bash
# Run 1 episode to verify everything works
python train.py --training.n_episodes 1

# Should complete without errors
```

---

## 🚀 HPC Cluster Setup

### SLURM Systems

Create `setup.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=setup_acc_rl
#SBATCH --time=00:30:00

# Load modules
module load python/3.10
module load cuda/11.8

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
```

Run with: `sbatch setup.sh`

### PBS Systems

Create `setup.pbs`:

```bash
#!/bin/bash
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00

# Load modules
module load python/3.10
module load cuda/11.8

# Setup
cd $PBS_O_WORKDIR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Setup complete!"
```

Run with: `qsub setup.pbs`

---

## 🔧 Troubleshooting

### Issue: "Python not found"

**Solution:**
```bash
# Use full path
/usr/bin/python3.10 -m venv venv

# Or install Python first
# See platform-specific instructions above
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure venv is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
python train.py --agent.batch_size 64

# Or use CPU
python train.py --training.cpu true
```

### Issue: "ImportError: No module named 'torch'"

**Solution:**
```bash
# Verify venv is active (should see (venv) in prompt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Permission denied"

**Linux/macOS:**
```bash
chmod +x train.py
chmod +x *.py
```

### Issue: "ImportError: cannot import name 'gymnasium'"

**Solution:**
```bash
pip install gymnasium --upgrade
```

---

## 📊 Environment Variables

### Useful Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1

# Temporary directory
export TMPDIR=/tmp

# Number of CPU threads
export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
```

### macOS Specific

```bash
# Use MPS (Metal Performance Shaders) if available
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Disable MPS if causing issues
export PYTORCH_MPS_ENABLED=0
```

---

## ✨ Post-Installation

### 1. Verify Setup

```bash
python -c "
import torch
import gymnasium
from rl_framework import ACCElegantEnvironment, DDPGAgent
print('✓ All modules loaded successfully')
"
```

### 2. Test Configuration System

```bash
python -c "
from config_manager import RLConfig
config = RLConfig.from_yaml('config.yaml')
print('✓ Configuration loaded successfully')
print(f'  Episodes: {config.training.n_episodes}')
print(f'  Seed: {config.training.seed}')
"
```

### 3. Run Test Episode

```bash
python train.py --training.n_episodes 1
```

### 4. Check Output

```bash
# Should see directories created:
ls -la results/
ls -la models/
ls -la logs/
```

---

## 🎓 Next Steps

1. **Review configuration**: Edit `config.yaml` for your needs
2. **Start training**: `python train.py`
3. **Monitor progress**: `tensorboard --logdir=results/`
4. **Adjust parameters**: Use CLI arguments or config file

## 📚 Additional Help

- `README.md`: Quick start guide
- `DEPLOYMENT_GUIDE.md`: Deployment instructions
- `config.yaml`: Configuration reference

---

**Setup Version**: 1.0
**Last Updated**: February 2026
**Status**: Production Ready ✅
