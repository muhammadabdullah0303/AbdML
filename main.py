import pandas as pd
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
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
            
    def get_metric(self, y_true, y_pred):
        if self.metric == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred.round())
        # Add other metrics as needed
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'")

    def LGBM(self, params):
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        train_scores = []
        oof_scores = []
        oof_predictions = np.zeros(len(self.y_train))
        test_preds = np.zeros((len(self.X_test), self.n_splits))

        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(self.X_train, self.y_train), desc="Training Folds", total=self.n_splits)):
            X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params) if self.problem_type == 'classification' else lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            y_train_pred = model.predict_proba(X_train)[:, 1] if self.problem_type == 'classification' else model.predict(X_train)
            y_val_pred = model.predict_proba(X_val)[:, 1] if self.problem_type == 'classification' else model.predict(X_val)

            oof_predictions[val_idx] = y_val_pred

            train_scores.append(self.get_metric(y_train, y_train_pred))
            oof_scores.append(self.get_metric(y_val, y_val_pred))

            test_preds[:, fold] = model.predict_proba(self.X_test)[:, 1] if self.problem_type == 'classification' else model.predict(self.X_test)

            print(f"Fold {fold + 1} - Train {self.metric.upper()}: {train_scores[-1]:.4f}, OOF {self.metric.upper()}: {oof_scores[-1]:.4f}")
            clear_output(wait=True)

        print(f"Overall Train {self.metric.upper()}: {np.mean(train_scores):.4f}")
        print(f"Overall OOF {self.metric.upper()}: {np.mean(oof_scores):.4f}")

        mean_test_preds = test_preds.mean(axis=1)
        return oof_predictions, mean_test_preds

def CAT(self, params):
    kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
    train_scores = []
    oof_scores = []
    oof_predictions = np.zeros(len(self.y_train))
    test_preds = np.zeros((len(self.X_test), self.n_splits))

    cat_features_indices = [self.X_train.columns.get_loc(col) for col in self.cat_features if col in self.X_train.columns]

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(self.X_train, self.y_train), desc="Training Folds", total=self.n_splits)):
        X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
        y_train, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

        X_train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
        X_val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

        model = CatBoostClassifier(**params) if self.problem_type == 'classification' else CatBoostRegressor(**params)
        model.fit(X_train_pool, eval_set=X_val_pool, verbose=False, early_stopping_rounds=200)

        if self.problem_type == 'classification':
            y_train_pred_proba = model.predict_proba(X_train_pool)[:, 1]
            y_val_pred_proba = model.predict_proba(X_val_pool)[:, 1]

            y_train_pred = (y_train_pred_proba > 0.5).astype(int)
            y_val_pred = (y_val_pred_proba > 0.5).astype(int)

            oof_predictions[val_idx] = y_val_pred_proba

            test_pool = Pool(self.X_test, cat_features=cat_features_indices)
            test_preds[:, fold] = model.predict_proba(test_pool)[:, 1]
        else:
            y_train_pred = model.predict(X_train_pool)
            y_val_pred = model.predict(X_val_pool)
            oof_predictions[val_idx] = y_val_pred

            test_pool = Pool(self.X_test, cat_features=cat_features_indices)
            test_preds[:, fold] = model.predict(test_pool)

        train_scores.append(self.get_metric(y_train, y_train_pred))
        oof_scores.append(self.get_metric(y_val, y_val_pred))

        print(f"Fold {fold + 1} - Train {self.metric.upper()}: {train_scores[-1]:.4f}, OOF {self.metric.upper()}: {oof_scores[-1]:.4f}")
        clear_output(wait=True)

    print(f"Overall Train {self.metric.upper()}: {np.mean(train_scores):.4f}")
    print(f"Overall OOF {self.metric.upper()}: {np.mean(oof_scores):.4f}")

    if self.problem_type == 'classification':
        mean_test_preds = (test_preds.mean(axis=1) > 0.5).astype(int)
    else:
        mean_test_preds = test_preds.mean(axis=1)
        
    return oof_predictions, mean_test_preds
