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
.
├── Project.ipynb                    # Main Colab notebook 
├── results/
│   ├── config.json                  # Dataset/experiment configuration
│   ├── classical_results.csv        # One-Class SVM results per category
│   ├── ae_results.csv               # Autoencoder results per category
│   ├── padim_results.csv            # PaDiM results per category
│   ├── final_results_all_methods.csv# Combined comparison table
│   ├── ae_model_<category>.pth      # Saved autoencoder weights (15 files) 
│   └── padim_maps_<category>.npy    # Saved PaDiM anomaly maps (15 files)
├── Final_Report.pdf
├── requirements.txt
└── README.md
```
* locate results/ and Project.ipynb in google drive
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


## Requirements

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
