# eh-high-altitude-screening

This repository accompanies the development and external validation of interpretable machine learning models for essential hypertension screening in high-altitude adults.

> Note: This repository contains code only. Participant-level data are not included due to institutional and privacy restrictions.

## Environment Setup

- Python version: `>=3.12`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Overview

- Task: Binary classification — predict the presence or absence of essential hypertension.
- Algorithms evaluated: Elastic Net, support vector machine, k-nearest neighbors, multilayer perceptron, random forest, ExtraTrees, XGBoost, LightGBM, AdaBoost, and others (12 models in total).
- Input: Structured routine laboratory variables (Excel files).
- Target label: `Group` (0 = healthy control; 1 = essential hypertension).
- Output: Trained model files (`.joblib`) and model performance results.

## Data Handling

- All code is structured to ensure clear separation between training and external validation.

For academic and research purposes only. No participant-level data are stored in this repository.

The model is intended to support screening prioritization and does not replace standardized blood-pressure measurement, diagnostic confirmation, or clinical judgement.

Additional code for data preprocessing and visualization will be uploaded soon before the paper is accepted.
