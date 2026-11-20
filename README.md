# Fashion-MNIST: Deep Learning Benchmark and Feature-Engineering Study

Classifying 28×28 grayscale clothing images into 10 categories, in two phases:
**Phase 1** benchmarks four neural networks, and **Phase 2** studies whether hand-engineered features and a better training recipe improve a CNN, with significance tests, multi-seed runs and calibration analysis.

## Problem statement

Given a small grayscale image of a clothing item, predict its class (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
Most classes are easy; the upper-body garments (T-shirt, Pullover, Coat, **Shirt**) look alike at 28×28, and **Shirt** is the hardest class for every model.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research, MIT license): 60,000 training and 10,000 test images.
Split used: **55,000 train / 5,000 validation / 10,000 test** (fixed seed). The validation set drives all design decisions; the test set is used only for reporting.
The official `idx` files are used (not the CSV re-export, whose rows and some pixel values differ). The dataset is not included in this repository.

## Phase 1: model benchmark (15 epochs, test set)

| Model | Test acc. | Macro F1 | Params (M) | Train time (s) |
|---|---|---|---|---|
| CNN | 93.96% | 0.939 | 0.585 | 126 |
| ResNet-18 style | 93.48% | 0.935 | 2.80 | 255 |
| ViT | 89.75% | 0.898 | 0.81 | 205 |
| VAE + linear head | 84.75% | 0.846 | 1.71 | 272 |

## Phase 2: feature engineering and training recipe

**Engineered features (885 dimensions):** HOG, fine HOG (4×4-pixel cells), LBP texture histograms, geometry/symmetry descriptors and Hu moments.

**2×2 ablation** (features × regularisation), 3 seeds each, same CNN branch and optimiser:

| Recipe | Test accuracy | NLL |
|---|---|---|
| A. CNN (baseline) | 93.80 ± 0.04% | 0.314 |
| A2. CNN + engineered features | 93.51 ± 0.35% | 0.326 |
| **B. CNN + light augmentation + label smoothing** | **94.35 ± 0.09%** | **0.191** |
| C. FusionNet (CNN + feature branch) | 92.82 ± 0.13% | 0.243 |

Final ensemble (B + C models, chosen on the validation set): **94.30%** test accuracy, macro-F1 0.943.

### Key findings
- **Light augmentation + label smoothing is the best recipe:** +0.55 points over the baseline (McNemar p = 0.010), with better calibration. A first attempt with heavy augmentation *lowered* accuracy (93.5% → 92.3%) because key details at 28×28 are only 1–2 pixels wide.
- **Engineered features help a classical classifier but not the CNN:** with an SVM, combined features beat raw pixels (89.0% vs 87.4% validation accuracy), but FusionNet was 1.5 points below the plain augmented CNN. Permutation importance shows it relied heavily on fine-HOG features (−6.78 points when shuffled).
- **Temperature scaling** reduces the final model's calibration error (ECE) from 0.043 to 0.010 without changing any prediction.
- **Remaining errors** are concentrated in Shirt vs T-shirt/Pullover/Coat.

### Methods and evaluation
Data augmentation (shift, flip, small cutout) · label smoothing · mixed-precision training · test-time (mirror) augmentation · seed ensembling · McNemar's test · expected calibration error (ECE) and temperature scaling · permutation feature importance · t-SNE and confusion-matrix analysis.

## Repository contents

| File | Description |
|---|---|
| `Fashion_MNIST_Benchmark_0079.ipynb` | Phase 1: four-model benchmark |
| `Fashion_MNIST_Final_Project_Phase2_v2.ipynb` | Phase 2: feature engineering, FusionNet, ablation, evaluation |
| `outputs/` | Result tables (CSV) and figures produced by the Phase 2 run |

## How to run

1. Open a notebook in Google Colab and set **Runtime → GPU**.
2. Upload the dataset zip (`archive.zip`, containing the official `*-ubyte` files) in the Files panel, or let the notebook download the data automatically.
3. Run all cells (Phase 2 takes about 45–60 minutes on a free GPU). Set `QUICK_TEST = True` for a ~3-minute smoke test.

**Requirements:** Python 3, PyTorch, torchvision, scikit-learn, scikit-image, scipy, pandas, numpy, matplotlib.

## Limitations
- 28×28 grayscale images limit fine detail.
- Hand-crafted features were designed for this dataset and largely duplicate what a CNN learns.
- Three seeds give only rough error bars.
- Phase 1 results come from a separate run (single seed).

## Acknowledgements
Fashion-MNIST: Xiao, Rasul and Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms* (2017).

## Author
**Prashantha Kumar S**  
Course: Deep Learning Techniques  
RV University
