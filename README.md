# Datasets, model weights, training logs, etc. will be uploaded later.

# DPGNet
## Start

- [Environment Setup](#-environment-setup)
- [Dataset](#-dataset)
- [Training](#-training)
- [Testing](#-testing)


## Environment Setup
Ensure your environment meets the following requirements:
-  Python 3.9
-  PyTorch 1.11
-  CUDA 11.3

Install dependencies:

```bash
cd DPGNet
conda create -n DPGNet python=3.9
conda activate DPGNet
sh install.sh
```

## Dataset

We use multiple datasets for training and evaluation:

- FF++
- DFDC
- DFDCP
- DFD
- CD1/CD2
- DF40

The dataset downloading and processing procedures can be referred to the implementation provided in [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) .


## Training
Make sure to modify the relevant configurations in the train.yaml file before training.

Start training with the following command:

```bash
python train.py 
```

## Testing
Make sure to modify the relevant configurations in the test.yaml file before testing.

To test the model, you can directly load our pre-trained weights and run a command like the following:

```bash
cd DPGNet
python test.py 
```
