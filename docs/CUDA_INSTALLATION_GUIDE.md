# PyTorch CUDA Installation Guide

## 🔍 Problem Diagnosis

If training shows CPU usage instead of GPU, it's usually because you have installed the CPU version of PyTorch.

## 🛠️ Solution

### 1. Check Current PyTorch Version

Run in Python:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

If `torch.cuda.is_available()` returns `False`, you have the CPU version installed.

### 2. Uninstall Current PyTorch

```bash
pip uninstall torch torchvision torchaudio
# or
conda uninstall pytorch torchvision torchaudio
```

### 3. Install CUDA Version of PyTorch

#### Method 1: Use Official Installation Command

Visit [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command.

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Method 2: Use conda Installation

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test GPU tensor creation
x = torch.randn(3, 3).cuda()
print(f"GPU tensor created: {x.device}")
```

## 🔧 Common Issues

### Issue 1: CUDA Version Mismatch
- Ensure PyTorch's CUDA version is compatible with system's CUDA version
- Check CUDA version displayed by `nvidia-smi` command

### Issue 2: Driver Problems
- Ensure NVIDIA drivers are up to date
- Restart system and retry

### Issue 3: Environment Issues
- If using conda environment, ensure correct environment is activated
- Check for multiple Python environment conflicts

## 📋 Checklist

- [ ] Uninstall CPU version of PyTorch
- [ ] Install CUDA version of PyTorch
- [ ] Verify CUDA availability
- [ ] Test GPU tensor creation
- [ ] Run training script to verify

## 🚀 Post-Installation Test

After installation, running the training script should show:

```
Using device: cuda
GPU: NVIDIA GeForce RTX 3080
GPU Memory: 10.0 GB
```

Instead of:

```
Using device: cpu
```

## 💡 Tips

1. **Recommend using conda**: conda usually handles CUDA dependencies better
2. **Version matching**: Ensure PyTorch CUDA version matches system CUDA version
3. **Driver updates**: Keep NVIDIA drivers up to date
4. **Environment isolation**: Use virtual environments to avoid package conflicts

## 🔗 Useful Links

- [PyTorch Installation Page](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility Table](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA Driver Download](https://www.nvidia.com/drivers/)