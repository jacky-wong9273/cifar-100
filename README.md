# 🖼️ CIFAR-100 Image Classification with ConvNeXt

## 📖 Introduction

This project implements a ConvNeXt-based convolutional neural network adapted for image classification on the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, achieving a **76.85% top-1 test accuracy** 🎯. CIFAR-100 consists of 60,000 32×32 colour images spanning 100 fine-grained classes, making it a challenging benchmark for evaluating modern vision architectures at low resolution.

The training pipeline features an aggressive yet stable data augmentation strategy—combining RandAugment, Mixup, and CutMix with curriculum scheduling—paired with AdamW optimisation and cosine-annealed learning rate warm-up to maximise performance from scratch without pre-training.

## 🚀 How to Run

### 📋 Prerequisites

- Python 3.8+
- PyTorch (with CUDA recommended)
- torchvision
- scikit-learn
- numpy, matplotlib, seaborn, tqdm

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy matplotlib seaborn tqdm
```

### 1️⃣ Download the Dataset

```bash
python main.py --dataset cifar100 --data_dir data
```

The pipeline will automatically download and organise CIFAR-100 into `data/raw/train` and `data/raw/val`.

### 2️⃣ Train the Model

Run the full training pipeline with default hyperparameters:

```bash
python main.py --dataset cifar100 --num_epochs 1000 --batch_size 512 --lr 4e-3 --weight_decay 5e-2
```

Key arguments:

| Argument                    | Default    | Description                                |
| --------------------------- | ---------- | ------------------------------------------ |
| `--dataset`                 | `cifar100` | Dataset (`cifar10` or `cifar100`)          |
| `--batch_size`              | `512`      | Training batch size                        |
| `--num_epochs`              | `1000`     | Maximum training epochs                    |
| `--lr`                      | `4e-3`     | Initial learning rate                      |
| `--weight_decay`            | `5e-2`     | AdamW weight decay                         |
| `--device`                  | `cuda`     | Device (`cuda` or `cpu`)                   |
| `--seed`                    | `3407`     | Random seed for reproducibility            |
| `--early_stopping_patience` | `1000`     | Epochs without improvement before stopping |

Checkpoints are saved to `results/models/` and training metrics to `results/results/`.

### 3️⃣ Evaluate

After training, the pipeline automatically evaluates the best checkpoint on the test set and produces classification reports and evaluation plots.

## 📁 Folder Structure

```
cifar-100/
├── main.py                          # Entry point – argument parsing, training loop, evaluation
├── scripts/
│   ├── model_architectures.py       # ConvNeXt model definition
│   ├── data_augmentation.py         # Runtime augmentation (Mixup, CutMix, curriculum scheduling)
│   ├── data_download.py             # CIFAR-10/100 download and extraction utilities
│   ├── train_utils.py               # Training/validation loops, optimizer, scheduler, checkpointing
│   └── evaluation_metrics.py        # Top-k accuracy, confusion matrix, PR curves, calibration plots
├── LICENSE
└── README.md
```

## 🧠 Model Architecture

The model is based on **ConvNeXt** (Liu et al., 2022), a pure-ConvNet architecture that modernises standard ResNets by incorporating design elements from Vision Transformers while remaining entirely convolutional.

### 🔧 ConvNeXt Adaptations for CIFAR-100

Because CIFAR-100 images are only 32×32 pixels—far smaller than the ImageNet inputs ConvNeXt was originally designed for—several architectural adjustments were made:

| Component          | Original ConvNeXt (ImageNet) | This Implementation (CIFAR-100) |
| ------------------ | ---------------------------- | ------------------------------- |
| Stem convolution   | 4×4, stride 4                | 4×4, stride 2, padding 1        |
| Depthwise kernel   | 7×7                          | 7×7 (preserved)                 |
| Stage depths       | [3, 3, 27, 3]                | [3, 3, 27, 3]                   |
| Channel dimensions | [96, 192, 384, 768]          | [96, 192, 384, 768]             |

### 🧱 Building Blocks

- **Depthwise separable convolutions** – Each ConvNeXt block applies a 7×7 depthwise convolution followed by two pointwise (1×1) linear layers with a GELU activation and Layer Scale.
- **Inverted bottleneck** – The hidden dimension expands to 4× the input channel count inside each block before projecting back, following the inverted bottleneck design.
- **DropPath (Stochastic Depth)** – A linearly increasing drop-path rate (0 → 0.4) across all 36 blocks provides strong regularisation.
- **Global average pooling** – Spatial features are aggregated via global average pooling before the final classification head.

### 🍳 Training Recipe

| Setting              | Value                                                                        |
| -------------------- | ---------------------------------------------------------------------------- |
| Optimiser            | AdamW (β₁=0.9, β₂=0.999)                                                     |
| Learning rate        | 4 × 10⁻³ with 20-epoch linear warm-up → cosine annealing to 5 × 10⁻⁷         |
| Weight decay         | 5 × 10⁻²                                                                     |
| Batch size           | 512                                                                          |
| Augmentation         | RandomCrop (32, pad 4 reflect), RandomHorizontalFlip, RandAugment (n=2, m=9) |
| Runtime augmentation | Mixup + CutMix (prob 0.95) with curriculum warm-up                           |
| Gradient clipping    | Max norm 1.0                                                                 |

### 📚 Reference

> Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. **"A ConvNet for the 2020s."** _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2022.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
