# PD_SWEDD_HC-Radiomics

This repository contains the official implementation for the study: 
**"Explainable MRI Radiomics for Differentiating Parkinson’s Disease, SWEDD, and Healthy Controls: A Multi-Class Machine Learning Framework."**

##  Project Overview
The goal of this study is to develop a robust machine learning framework based on MRI radiomics to differentiate between PD patients, SWEDD subjects, and healthy controls. We prioritize model interpretability through SHAP analysis and provide spatial feature mapping to validate the biological plausibility of our radiomic markers.

##  Key Features
- **70/30 Train-Test Split**: Ensures unbiased performance evaluation on a strictly independent test set.
- **Sliding Window Mapping**: Implementation of a 3D sliding window methodology to generate voxel-wise feature maps.
- **Interpretability**: Integration of SHAP analysis to identify and rank the 14 most influential radiomic features.
- **Reproducibility**: Fixed random seeds (e.g., seed=42) for all data splitting and model training procedures.

##  Repository Structure
- **feature_engineering**: Contains scripts for local radiomic feature extraction via the 3D sliding window method and the consensus-based selection (p-value, mRMR, and LASSO) performed on the training cohort.
- **modeling_and_prediction**: Includes the randomized 70/30 train-test split, hyperparameter tuning for all five evaluated models, and the final model locking for XGBoost.
- **interpretability_and_mapping**: Scripts for SHAP analysis and the generation of voxel-wise spatial mapping (as presented in Figure 8) to validate neuroanatomical consistency.
- **requirements.txt**: List of Python dependencies required to run the analysis

##  Getting Started
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yeshiva0304/PD_SWEDD_HC-Radiomics.git](https://github.com/yeshiva0304/PD_SWEDD_HC-Radiomics.git)
