<img src="https://img.shields.io/badge/Python-white?logo=Python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/pandas-white?logo=pandas&logoColor=250458" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/polars-white?logo=polars&logoColor=blue" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/NumPy-white?logo=numpy&logoColor=013243" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/PyTorch-white?logo=PyTorch" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Scikit_learn-white?logo=scikitlearn&logoColor=F7931E" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/mlflow-white?logo=mlflow&logoColor=blue" style="height: 25px; width: auto;">


# Skin Cancer Classification Using Ensemble of Image-Based and Tabular-Based Algorithms

## Overview

This repository contains the code used for competing in the "ISIC 2024 - Skin Cancer Detection with 3D-TBP" Kaggle competition, which focuses on classifying malignant skin lesions from single-lesion crops of 3D total body photos (TBP). The images resemble close-up smartphone photos and are used for improving early skin cancer detection in telehealth settings.

## Challenges

1. **Image Quality**: The dataset includes low-resolution images, similar to those from smartphone cameras, affecting model performance.
2. **Class Imbalance**: The dataset is highly imbalanced, with over 400,000 benign samples and only 393 malignant samples, creating difficulties in model training and evaluation.

## Approach

Given the dataset's characteristics, the approach involved using several PyTorch models for inference on both training and test datasets. Due to the low quality of the images and their similarity to cell phone photos, complex network architectures were avoided. Instead, simpler models were chosen to retain essential information without overfitting.

- **EfficientNet-B0**: A less complex version of the EfficientNet series.
- **GhostNet**: Lightweight and designed for efficient computation.
- **MobileNet**: Known for its small size and efficiency.
- **MixNet-S**: Designed for both efficiency and performance with moderate complexity.

These models' predictions were used as additional features alongside the metadata for training models like **LightGBM and CatBoost**. This combination aimed to leverage both image and metadata features for improved classification performance.

## Notebooks

### Notebook 0: Exploratory Data Analysis (EDA)

This notebook performs exploratory data analysis of the metadata, which includes various patient and image features. Key features analyzed include:
- **tbp_lv_H**: Hue inside the lesion, representing color intensity (typical values range from 25 for red to 75 for brown).
- **tbp_lv_areaMM2**: Area of the lesion in square millimeters.

The analysis helps in understanding the distribution and relationships of these features within the dataset.

### Notebook 1: Data Preprocessing and Feature Engineering

This notebook focuses on preprocessing the metadata, including encoding and feature engineering. It also involves optimizing parameters for data handling, such as under-sampling and over-sampling using SMOTE to address class imbalance. The primary model used in this notebook is LightGBM, and experiments are tracked using MLflow to monitor performance and hyperparameter tuning.

### Notebook 2: Integration of Image-Based Features and Final Model Training

This notebook integrates the predictions from image-based models (EfficientNet-B0, GhostNet, MobileNet, MixNet-S) with the metadata features. Predictions for both training and test datasets were obtained using separate notebooks on Kaggle and through the `main.py` script included in the repository. For final cross-validation and prediction, an ensemble of CatBoost and LightGBM models is employed. Additionally, deeper parameter tuning was conducted using Optuna to optimize model performance.


