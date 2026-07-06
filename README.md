# Manufacturing Defect Detection using MVTec AD Dataset

**Course:** CSC 482 — Applied Image Analysis
**Student:** Krishnarjun Lakshminarayanan  

https://drive.google.com/drive/folders/15owCcHxsJkd-gNGoeuiBjlefK63luc1h?usp=sharing

---

## Project Overview

This project implements and compares three automated visual inspection methods for detecting manufacturing defects in industrial products using the **MVTec Anomaly Detection (MVTec AD)** dataset. All three methods are **unsupervised** — they train exclusively on normal (defect-free) images and detect anomalies as deviations from the learned normal distribution. No defect labels are required during training, which mirrors real-world factory conditions.

### Methods Implemented

| Method | Approach | Mean ROC AUC | Inference Time |
|--------|----------|:------------:|:--------------:|
| One-Class SVM | Classical: LBP + GLCM + Canny/Sobel + PCA | 0.6468 | 88.2 ms |
| Convolutional Autoencoder | Deep Learning: MSE reconstruction error | 0.5625 | 1.4 ms |
| **PaDiM** | **Deep Learning: ResNet18 + Patch Gaussians + Mahalanobis** | **0.9262** | **10.3 ms** |

---

## Repository Structure

```
Project.ipynb                    # Main Colab notebook (all 5 parts)
README.md                        # This file

results/                         # Saved to Google Drive during execution
    config.json                  # Dataset/experiment configuration
    classical_results.csv        # One-Class SVM results per category
    ae_results.csv               # Autoencoder results per category
    padim_results.csv            # PaDiM results per category
    final_results_all_methods.csv# Combined comparison table
    ae_model_<category>.pth      # Saved autoencoder weights (15 files)
    padim_maps_<category>.npy    # Saved PaDiM anomaly maps (15 files)
```

---

## Dataset

**MVTec Anomaly Detection (MVTec AD)**  
- **Source:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)  
- **Kaggle:** [https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library/input](https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library/input)
- **15 Categories:** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
- **5,354 total images** | **3,629 training (normal only)** | **1,725 test (normal + defect)**

### Expected Folder Structure

```
MyDrive/CSC482-Project/mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/          # Normal training images
│   ├── test/good/           # Normal test images
│   ├── test/broken_large/   # Defective test images
│   ├── test/broken_small/
│   ├── test/contamination/
│   └── ground_truth/        # Binary mask images
│       ├── broken_large/
│       ├── broken_small/
│       └── contamination/
├── cable/
│   └── ...
└── [13 more categories]
```

---

## Environment Setup

### Google Colab (Recommended)

This notebook is designed to run on **Google Colab** with a GPU runtime.

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → GPU (T4 or A100)**
3. Upload `Project.ipynb` via **File → Upload notebook**
4. Mount Google Drive and ensure the dataset is at:
   ```
   /content/drive/MyDrive/CSC482-Project/mvtec_anomaly_detection/
   ```
5. Run all cells in order (Part 1 → Part 5)

### Required Libraries

All dependencies are installed automatically in Cell 1:

```bash
pip install scikit-image opencv-python-headless matplotlib seaborn scikit-learn tqdm Pillow
```

PyTorch and torchvision are pre-installed on Colab. Additional imports used:

```python
# Core
import torch, torchvision
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2, json, os, time

# Classical ML
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.color import rgb2gray
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# Deep Learning
import torch.nn as nn, torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# PaDiM
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
```

---

## Notebook Structure

The notebook is organized into **5 parts** with **54 code cells** total:

### Part 1 — Setup & Data Loading (Cells 0–13)
| Cell | Description |
|------|-------------|
| 0 | Install libraries |
| 1 | Import all dependencies, set random seed (42) |
| 2 | Mount Google Drive |
| 3 | Set `DATASET_ROOT` path, verify 15 categories |
| 4 | `explore_category()` — count images per split/defect type |
| 5 | Remount Drive |
| 6 | Build and print dataset summary table |
| 7 | Plot distribution charts (bar + pie) |
| 8 | `load_image()` and `get_sample_images()` utilities |
| 9 | Visualize sample images per category |
| 10 | Check native image resolutions per category |
| 11 | `MVTecDataset` class — unified data loader with mask support |
| 12 | `visualize_with_mask()` — show defect + ground truth overlay |
| 13 | Save `CONFIG` dict to Drive as `config.json` |

### Part 2 — Classical Baseline (Cells 14–23)
| Cell | Description |
|------|-------------|
| 14 | Import classical ML libraries |
| 15 | Feature extractors: `extract_lbp_features()`, `extract_glcm_features()`, `extract_edge_features()`, `extract_statistical_features()`, `extract_all_features()` |
| 16 | `build_feature_matrix()` — build feature arrays for train/test splits |
| 17 | `train_ocsvm()` — StandardScaler → PCA → OneClassSVM pipeline |
| 18 | `evaluate_predictions()` — ROC AUC, Average Precision, ROC/PR curves |
| 19 | `plot_score_distribution()` — histogram + boxplot of anomaly scores |
| 20 | Full pipeline across all 15 categories |
| 21 | Build results DataFrame, save `classical_results.csv` |
| 22 | Visualize results across categories (horizontal bar chart) |
| 23 | `qualitative_analysis()` — true positives, false negatives, false positives |

### Part 3 — Convolutional Autoencoder (Cells 24–34)
| Cell | Description |
|------|-------------|
| 24 | Import PyTorch, detect GPU (`NVIDIA A100-SXM4-80GB`) |
| 25 | `MVTecTorchDataset` — PyTorch Dataset wrapper with ImageNet normalization |
| 26 | `ConvAutoencoder` — 4-block encoder/decoder (562,467 params) |
| 27 | `train_autoencoder()` — Adam optimizer, ReduceLROnPlateau, early stopping |
| 28 | `plot_training_history()` — loss curve with best epoch marker |
| 29 | `get_anomaly_scores_ae()` — compute reconstruction errors on test set |
| 30 | Evaluate AE on bottle category |
| 31 | `visualize_reconstructions()` — original vs reconstructed side-by-side |
| 32 | `visualize_error_heatmaps()` — per-pixel error maps vs ground truth |
| 33 | Full AE pipeline across all 15 categories, save `.pth` model files |
| 34 | Build AE results DataFrame, compare vs classical |

### Part 4 — PaDiM (Cells 35–44)
| Cell | Description |
|------|-------------|
| 35 | Import PaDiM-specific libraries |
| 36 | `PaDiMFeatureExtractor` — frozen ResNet18, extract layer1+2+3 features (448-dim) |
| 37 | `extract_train_features()` — extract + SparseRandomProjection (448→100) |
| 38 | `fit_gaussian_per_patch()` — multivariate Gaussian per 64×64 patch position |
| 39 | `compute_anomaly_maps()` — Mahalanobis distance maps, Gaussian smoothing, upsampling |
| 40 | Evaluate PaDiM on bottle |
| 41 | `visualize_padim_maps()` — original + heatmap + overlay + GT mask |
| 42 | `compute_localization_metrics()` — IoU and Dice coefficient |
| 43 | Full PaDiM pipeline across all 15 categories, save `.npy` anomaly maps |
| 44 | Build PaDiM results DataFrame, 3-method comparison chart |

### Part 5 — Final Evaluation (Cells 45–53)
| Cell | Description |
|------|-------------|
| 45 | Load all saved results from Drive |
| 46 | Radar chart comparing all 3 methods across 15 categories |
| 47 | `benchmark_inference_time()` — 10-run timing benchmark |
| 48 | `plot_performance_heatmap()` — ROC AUC heatmap (methods × categories) |
| 49 | `plot_localization_metrics()` — IoU & Dice bar charts |
| 50 | `plot_error_analysis()` — best/worst 5 categories per method |
| 51 | `plot_final_dashboard()` — 6-panel comprehensive summary figure |
| 52 | `print_final_report()` — full text summary of all results |

---

## Hardware Used

- **Platform:** Google Colab
- **GPU:** NVIDIA A100-SXM4-80GB
- **GPU Memory:** 85.1 GB
- **Storage:** Google Drive (MyDrive/CSC482-Project/) 

---

## References

1. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection**. CVPR 2019.

2. Defard, T., Setkov, A., Loesch, A., & Audigier, R. (2021). **PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization**. ICPR 2021.

3. Bergmann, P., Löwe, S., Fauser, M., Sattlegger, D., & Steger, C. (2018). **Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders**. VISAPP 2019.

---

## License

This project is submitted as coursework for CSC 482 at the university. The MVTec AD dataset is used under the MVTec Research License for non-commercial academic use.
