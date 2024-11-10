
import pandas as pd
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import *
from IPython.display import clear_output

SEED = 42

class AbdBase:
    
    model_name = ["LGBM", "CAT", "XGB"]
    metrics = ["roc_auc", "accuracy", "f1", "precision", "recall", 'rmse']
    regression_metrics = ["mae", "r2"]
    problem_types = ["classification", "regression"]
    cv_types = ['SKF', 'KF', 'GKF', 'GSKF']
    
    def __init__(self, train_data, test_data=None, target_column=None,
                 problem_type="classification", metric="roc_auc", seed=SEED,
                 n_splits=5, cat_features=None, num_classes=None, prob=True, 
                 early_stop=False, test_prob=False, fold_type='SKF'):
        self.train_data = train_data
        self.test_data = test_data
        self.target_column = target_column
        self.problem_type = problem_type
        self.metric = metric
        self.seed = seed
        self.n_splits = n_splits
        self.cat_features = cat_features if cat_features else []
        self.num_classes = num_classes
        self.prob = prob 
        self.test_prob = test_prob
        self.early_stop = early_stop
        self.fold_type = fold_type

        self._validate_input()
        self.checkTarget()
        
        self.X_train = self.train_data.drop(self.target_column, axis=1)
        self.y_train = self.train_data[self.target_column]
        self.X_test = self.test_data if self.test_data is not None else None
        
        self._display_initial_info()

    def checkTarget(self):
        if self.train_data[self.target_column].dtype == 'object':
            raise ValueError('Encode Target First')
        
    def _display_initial_info(self):
        print("Available Models:", ", ".join(self.model_name))
        print("Available Metrics:", ", ".join(self.metrics))
        print("Available Problem Types:", ", ".join(self.problem_types))
        print("Available Fold Types:", ", ".join(self.cv_types))
        print(f"Problem Type Selected: {self.problem_type}")
        print(f"Metric Selected: {self.metric}")
        print(f"Fold Type Selected: {self.fold_type}")
        print(f"Calculate Train Probabilities: {self.prob}")
        print(f"Calculate Test Probabilities: {self.test_prob}")
        print(f"Early Stopping: {self.early_stop}")

    def _validate_input(self):
        if not isinstance(self.train_data, pd.DataFrame):
            raise ValueError("Training data must be a pandas DataFrame.")
        if self.test_data is not None and not isinstance(self.test_data, pd.DataFrame):
            raise ValueError("Test data must be a pandas DataFrame.")
        if self.target_column not in self.train_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the training dataset.")
        if self.problem_type not in self.problem_types:
            raise ValueError("Invalid problem type. Choose either 'classification' or 'regression'.")
        if self.metric not in self.metrics and self.metric not in self.regression_metrics:
            raise ValueError("Invalid metric. Choose from available metrics.")
        if not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValueError("n_splits must be an integer greater than 1.")
        if self.fold_type not in self.cv_types:
            raise ValueError(f"Invalid fold type. Choose from {self.cv_types}.")

    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, 
        mean_absolute_error, r2_score, mean_squared_error
    )

    def get_metric(self, y_true, y_pred):
        if self.metric == 'roc_auc':
            return roc_auc_score(y_true, y_pred, multi_class="ovr" if self.num_classes > 2 else None)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred.round())
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred.round(), average='weighted') if self.num_classes > 2 else f1_score(y_true, y_pred.round())
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred.round(), average='weighted') if self.num_classes > 2 else precision_score(y_true, y_pred.round())
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred.round(), average='weighted') if self.num_classes > 2 else recall_score(y_true, y_pred.round())
        elif self.metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == 'r2':
            return r2_score(y_true, y_pred)
        elif self.metric == 'rmse':
            return mean_squared_error(y_true, y_pred, squared=False)
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'")

    def Train_ML(self, params, model_name, e_stop=50):
        print(f"The Val of EarlyStopping is {e_stop}")
        if self.metric not in self.metrics:
            raise ValueError(f"Metric '{self.metric}' is not supported. Choose from Given Metrics.")
        if self.problem_type not in self.problem_types:
            raise ValueError(f"Problem type '{self.problem_type}' is not supported. Choose from: 'classification', 'regression'.")

        if self.fold_type == 'SKF':
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.fold_type == 'KF':
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.fold_type == 'GKF':
            kfold = GroupKFold(n_splits=self.n_splits)
        elif self.fold_type == 'GSKF':
            raise NotImplementedError("Group Stratified KFold not implemented yet.")

        train_scores = []
        oof_scores = []
        oof_predictions = np.zeros((len(self.y_train), self.num_classes)) if self.num_classes > 2 else np.zeros(len(self.y_train))
        test_preds = (
            None if self.X_test is None else
            np.zeros((len(self.X_test), self.n_splits, self.num_classes)) if self.num_classes > 2 else
            np.zeros((len(self.X_test), self.n_splits))
        )

        cat_features_indices = [self.X_train.columns.get_loc(col) for col in self.cat_features] if model_name == 'CAT' else None

        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(self.X_train, self.y_train), desc="Training Folds", total=self.n_splits)):
            X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            if model_name == 'LGBM':
                model = lgb.LGBMClassifier(**params, random_state=self.seed, verbose=-1) if self.problem_type == 'classification' else lgb.LGBMRegressor(**params, random_state=self.seed, verbose=-1)
            elif model_name == 'CAT':
                train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features_indices)
                val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features_indices)
                model = CatBoostClassifier(**params, random_state=self.seed, verbose=0) if self.problem_type == 'classification' else CatBoostRegressor(**params, random_state=self.seed, verbose=0)
            else:
                raise ValueError("model_name must be 'LGBM' or 'CAT'.")

            callbacks = [early_stopping(stopping_rounds=e_stop, verbose=False)]
            if model_name == 'LGBM':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],callbacks=callbacks if self.early_stop else None)
            elif model_name == 'CAT':
                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=e_stop if self.early_stop else None)

            if self.problem_type == 'classification':
                y_train_pred = model.predict_proba(X_train)[:, 1] if self.prob else model.predict(X_train)
                y_val_pred = model.predict_proba(X_val)[:, 1] if self.prob else model.predict(X_val)
            else:
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

            oof_predictions[val_idx] = y_val_pred

            if self.metric == "accuracy":
                train_scores.append(accuracy_score(y_train, np.argmax(y_train_pred, axis=1) if self.num_classes > 2 else (y_train_pred > 0.5).astype(int)))
                oof_scores.append(accuracy_score(y_val, np.argmax(y_val_pred, axis=1) if self.num_classes > 2 else (y_val_pred > 0.5).astype(int)))
            elif self.metric == "roc_auc":
                train_scores.append(roc_auc_score(y_train, y_train_pred, multi_class="ovr" if self.num_classes > 2 else None))
                oof_scores.append(roc_auc_score(y_val, y_val_pred, multi_class="ovr" if self.num_classes > 2 else None))
            else:
                train_scores.append(self.get_metric(y_train, y_train_pred))
                oof_scores.append(self.get_metric(y_val, y_val_pred))
    
            if self.X_test is not None:
                if self.problem_type == 'classification':
                    test_preds[:, fold] = model.predict_proba(self.X_test)[:, 1] if self.test_prob else model.predict(self.X_test)
                else:
                    test_preds[:, fold] = model.predict(self.X_test)

            print(f"Fold {fold + 1} - Train {self.metric.upper()}: {train_scores[-1]:.4f}, OOF {self.metric.upper()}: {oof_scores[-1]:.4f}")
            clear_output(wait=True)

        print(f"Overall Train {self.metric.upper()}: {np.mean(train_scores):.4f}")
        print(f"Overall OOF {self.metric.upper()}: {np.mean(oof_scores):.4f}")

        mean_test_preds = test_preds.mean(axis=1) if self.X_test is not None else None
        return oof_predictions, mean_test_preds
