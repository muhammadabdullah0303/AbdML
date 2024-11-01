

import pandas as pd
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from IPython.display import clear_output

SEED = 42

class AbdBase:
    
    def __init__(self, train_data, test_data, target_column, problem_type, metric, seed=SEED, n_splits=5, cat_features=None):
        self.train_data = train_data
        self.test_data = test_data
        self.target_column = target_column
        self.problem_type = problem_type
        self.metric = metric
        self.seed = seed
        self.n_splits = n_splits
        self.cat_features = cat_features if cat_features else []

        self._validate_input()

        self.X_train = self.train_data.drop(self.target_column, axis=1)
        self.y_train = self.train_data[self.target_column]
        self.X_test = self.test_data

    def _validate_input(self):
        if not isinstance(self.train_data, pd.DataFrame):
            raise ValueError("Training data must be a pandas DataFrame.")
        if not isinstance(self.test_data, pd.DataFrame):
            raise ValueError("Test data must be a pandas DataFrame.")
        if self.target_column not in self.train_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the training dataset.")
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError("Problem type must be either 'classification' or 'regression'.")
        if not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValueError("n_splits must be an integer greater than 1.")

    def LGBM(self, params):
        SKFold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        train_scores = []
        oof_scores = []
        oof_predictions = np.zeros(len(self.y_train))
        test_preds = np.zeros((len(self.X_test), self.n_splits))

        for fold, (train_idx, val_idx) in enumerate(tqdm(SKFold.split(self.X_train, self.y_train), desc="Training Folds", total=self.n_splits)):
            X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            oof_predictions[val_idx] = y_val_pred

            train_auc = roc_auc_score(y_train, y_train_pred)
            val_auc = roc_auc_score(y_val, y_val_pred)

            train_scores.append(train_auc)
            oof_scores.append(val_auc)

            test_preds[:, fold] = model.predict_proba(self.X_test)[:, 1]

            print(f"Fold {fold + 1} - Train AUC: {train_auc:.4f}, OOF AUC: {val_auc:.4f}")
            clear_output(wait=True)

        print(f"Overall Train AUC: {np.mean(train_scores):.4f}")
        print(f"Overall OOF AUC: {np.mean(oof_scores):.4f}")

        mean_test_preds = np.round(test_preds.mean(axis=1)).astype(int)
        return oof_predictions, mean_test_preds

    def CAT(self, params):
        SKFold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        train_scores = []
        oof_scores = []
        oof_predictions = np.zeros(len(self.y_train))
        test_preds = np.zeros((len(self.X_test), self.n_splits))

        cat_features_indices = [self.X_train.columns.get_loc(col) for col in self.cat_features if col in self.X_train.columns]

        for fold, (train_idx, val_idx) in enumerate(tqdm(SKFold.split(self.X_train, self.y_train), desc="Training Folds", total=self.n_splits)):
            X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            X_train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
            X_val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

            model = CatBoostClassifier(**params)
            model.fit(X=X_train_pool, eval_set=X_val_pool, verbose=False, early_stopping_rounds=200)

            y_train_pred = model.predict(X_train_pool)
            y_val_pred = model.predict(X_val_pool)

            oof_predictions[val_idx] = y_val_pred

            train_auc = roc_auc_score(y_train, y_train_pred)
            val_auc = roc_auc_score(y_val, y_val_pred)

            train_scores.append(train_auc)
            oof_scores.append(val_auc)

            test_pool = Pool(self.X_test, cat_features=cat_features_indices)
            test_preds[:, fold] = model.predict_proba(test_pool)[:, 1]

            print(f"Fold {fold + 1} - Train AUC: {train_auc:.4f}, OOF AUC: {val_auc:.4f}")
            clear_output(wait=True)

        print(f"Overall Train AUC: {np.mean(train_scores):.4f}")
        print(f"Overall OOF AUC: {np.mean(oof_scores):.4f}")

        mean_test_preds = np.round(test_preds.mean(axis=1)).astype(int)
        return oof_predictions, mean_test_preds
