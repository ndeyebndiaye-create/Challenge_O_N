# 🧠 Brain Tumor Classification from MRI Scans

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> Deep learning pipeline for brain tumor detection and classification using MRI images — from raw data cleaning to transfer learning with performance comparison.

---

## 📌 Project Overview

Brain tumors are abnormal cell growths within the brain. Due to the rigid structure of the skull, any abnormal growth can increase intracranial pressure and potentially lead to severe neurological damage. **Early detection and accurate classification are critical** for selecting appropriate treatment strategies and improving patient outcomes.

This project builds a complete end-to-end deep learning pipeline to classify brain MRI scans into 4 categories:

| Class | Description |
|---|---|
| 🔴 Glioma | Tumor originating from glial cells |
| 🟠 Meningioma | Tumor arising from the meninges |
| 🟡 Pituitary | Tumor in the pituitary gland |
| 🟢 No Tumor | Healthy brain scan |

---

## 📂 Dataset

**Source:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Testing)

```
dataset/
├── Training/
│   ├── glioma/        (1,400 images)
│   ├── meningioma/    (1,400 images)
│   ├── pituitary/     (1,400 images)
│   └── notumor/       (1,400 images)
└── Testing/
    ├── glioma/        (300 images)
    ├── meningioma/    (300 images)
    ├── pituitary/     (300 images)
    └── notumor/       (300 images)
```

### ⚠️ Real-World Data Challenges

This dataset reflects real clinical imaging conditions:

- **Noisy images** — MRI artifacts, varying orientations, different slice planes
- **Blurry or poorly framed** scans
- **Inconsistent resolutions** across images
- **Class imbalance** between tumor and non-tumor samples

These challenges are treated explicitly in the preprocessing pipeline.

---

## 🛠️ Pipeline

### Step 1 — Data Collection & Exploration
- Download dataset from Kaggle
- Analyze class distribution
- Identify and visualize problematic images (blurry, corrupt, artifacts)

### Step 2 — Cleaning & Preprocessing
- Remove corrupted or overly blurry images
- Resize all images to **224×224**
- Normalize pixel values to **[0, 1]**
- Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance MRI contrast

### Step 3 — Handling Class Imbalance
- **Data augmentation** on underrepresented classes (rotation, flip, zoom, shear)
- Alternatively: use **class weights** in the loss function

### Step 4 — Modeling

| Model | Description |
|---|---|
| 🔵 Baseline CNN | Custom architecture trained from scratch |
| 🟣 MobileNetV2 | Lightweight transfer learning |
| 🔴 ResNet50 | Deeper transfer learning for comparison |

Results are compared **before and after data cleaning** to quantify the impact of preprocessing.

### Step 5 — Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve (one-vs-rest)

---

## 📊 Results

> ⚙️ *Results will be updated as experiments are completed.*

| Model | Accuracy | F1-Score | Notes |
|---|---|---|---|
| Baseline CNN (raw data) | — | — | No preprocessing |
| Baseline CNN (clean data) | — | — | After cleaning |
| MobileNetV2 | — | — | Transfer learning |
| ResNet50 | — | — | Transfer learning |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python kaggle
```

### Clone & Run

```bash
git clone https://github.com/your-username/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

Download the dataset:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d dataset/
```

Run the pipeline:

```bash
python src/preprocess.py     # Cleaning & preprocessing
python src/train_baseline.py # Train baseline CNN
python src/train_transfer.py # Train MobileNetV2 / ResNet50
python src/evaluate.py       # Generate metrics & plots
```

---

## 📁 Repository Structure

```
brain-tumor-mri-classification/
├── dataset/                  # Raw and cleaned data (not tracked by git)
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   └── 04_transfer_learning.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_transfer.py
│   └── evaluate.py
├── models/                   # Saved model weights
├── outputs/                  # Confusion matrices, ROC curves, plots
├── requirements.txt
└── README.md
```

---

## 🧪 Key Technical Choices

**Why CLAHE?**  
Standard histogram equalization can over-amplify noise. CLAHE operates on local regions, making it ideal for MRI scans where contrast varies significantly across the image.

**Why compare before/after cleaning?**  
Demonstrating the measurable impact of proper preprocessing is central to this project — not just getting good results, but understanding *why* they improve.

**Why MobileNetV2 and ResNet50?**  
Both are proven ImageNet backbones. MobileNetV2 is fast and lightweight; ResNet50 is deeper and more expressive. Comparing both gives insight into the accuracy/efficiency tradeoff on medical imaging.

---

## 📚 References

- [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [CLAHE — OpenCV Documentation](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [MobileNetV2 — Howard et al., 2018](https://arxiv.org/abs/1801.04381)
- [ResNet — He et al., 2015](https://arxiv.org/abs/1512.03385)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> 💡 *This project was built as a complete data science portfolio piece demonstrating real-world medical image preprocessing, class imbalance handling, and deep learning model comparison.*
