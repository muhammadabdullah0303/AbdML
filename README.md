# AbdBase Class

## Overview

`AbdBase` is a versatile machine learning utility class designed for various tasks, including classification and regression. It integrates popular models, such as LGBM, XGB, and TabNet, and provides tools for cross-validation, feature engineering, and model evaluation with multiple metrics.

## Features

- **Model Support:** LGBM, CAT, XGB, Voting, TABNET
- **Metrics:** Supports a wide range of evaluation metrics like accuracy, ROC AUC, F1, MAE, RMSE, etc.
- **Cross-validation:** Multiple cross-validation techniques including StratifiedKFold (SKF), KFold (KF), and GroupKFold (GKF).
- **Problem Type:** Supports both classification and regression tasks.
- **Feature Engineering:** Options for target encoding, one-hot encoding, and TF-IDF for multi-column text data.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

```python
from abd_base import AbdBase

# Initialize the AbdBase class
model = AbdBase(train_data=train_data, test_data=test_data, target_column="target")
model.fit()
```

## Parameters

- `train_data`: Training dataset (required).
- `test_data`: Test dataset (optional).
- `target_column`: Column name for the target variable (optional).
- `problem_type`: Type of problem, either "classification" or "regression".
- `metric`: Metric for model evaluation.
- `n_splits`: Number of splits for cross-validation.
- `cat_features`: List of categorical features.
- `gpu`: Whether to use GPU acceleration (optional).
