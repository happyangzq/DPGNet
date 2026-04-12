<div align="center">

# DPGNet: Leveraging Unlabeled Data from Unknown Sources via Dual-Path Guidance for Deepfake Face Detection

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2508.09022)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 1.11](https://img.shields.io/badge/pytorch-1.11-%237732a8)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch Implementation of DPGNet**

</div>

## 📢 News & Updates
- [ ] **[TODO]** Datasets, model weights, and training logs will be released soon. Stay tuned!
- [x] Initial release of the training and testing code for **DPGNet**.

<details open>
<summary><b>📖 Table of Contents</b></summary>

- [Environment Setup](#️-environment-setup)
- [Data Preparation](#️-data-preparation)
- [Getting Started](#-getting-started)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

</details>

---

## 🛠️ Environment Setup

Ensure your environment meets the following prerequisites:
- **Python** $\ge$ 3.9
- **PyTorch** $\ge$ 1.11
- **CUDA** $\ge$ 11.3

**Step-by-step installation:**

```bash
# Clone the repository
git clone https://github.com/YourUsername/DPGNet.git
cd DPGNet

# Create and activate the conda environment
conda create -n DPGNet python=3.9 -y
conda activate DPGNet

# Install dependencies
sh install.sh
```

## 🗂️ Data Preparation

We conduct extensive evaluations across multiple deepfake detection benchmark datasets:

- **FF++** (FaceForensics++)
- **DFDC** (DeepFake Detection Challenge)
- **DFDCP** (DeepFake Detection Challenge Preview)
- **DFD** (DeepFakeDetection)
- **CD1/CD2** (Celeb-DF-v1 / Celeb-DF-v2)
- **DF40** (DeepFake-40)

> **Note:** For standardized dataset downloading and preprocessing procedures, please refer to the excellent pipeline provided by [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).

## 🚀 Getting Started

### Training

Before initiating the training process, please configure the necessary hyperparameters and dataset paths in `train.yaml`.

Start training with the following command:

```bash
python train.py 
```

### Evaluation

Make sure to modify the relevant configurations in `test.yaml` before testing. To evaluate the model, you can directly load our pre-trained weights (to be released) and run:

```bash
python test.py 
```


## 🙏 Acknowledgements

This repository borrows dataset preprocessing components from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). We sincerely thank the authors for their great contribution to the community.
