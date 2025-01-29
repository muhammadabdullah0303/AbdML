# V_1.3 # Include TabNET + OPTUNA Tunning
# !pip install -qq pytorch_tabnet
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
# from pytorch_tabnet.tab_model import TabNetRegressor,TabNetClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.model_selection import *
from sklearn.metrics import *
from IPython.display import clear_output
from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from colorama import Fore
from nltk.corpus import stopwords
import nltk
import string
import optuna
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter('ignore')
import optuna
from typing import Dict, Any, Optional, Union, Tuple
from colorama import Fore
import logging
import pandas.api.types
from lifelines.utils import concordance_index
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pandas as pd   # For working with data
import plotly.graph_objects as go   # For creating interactive plots

class ParticipantVisibleError(Exception):
    pass


SEED = 42

class AbdBase:
    
    model_name = ["LGBM", "CAT", "XGB","Voting",'TABNET','Ridge',"LR"]
    metrics = ["roc_auc", "accuracy", "f1", "precision", "recall", 'rmse','wmae',"rmsle","mae", "r2",'mse',
              'mape',"custom"]
    problem_types = ["classification", "regression"]
    cv_types = ['SKF', 'KF', 'GKF', 'GSKF',"RKF"]
    current_version = ['V_1.3']
    
    def __init__(self, train_data, test_data=None, target_column=None,tf_vec=False,gpu=False,numpy_data=False,handle_date=False,
                 ordinal_encoder = False,
                 problem_type="classification", metric="roc_auc", seed=SEED,ohe_fe=False,label_encode=False,target_encode=False,
                 n_splits=5, cat_features=None, num_classes=None, prob=False,stat_fe = None,logger: Optional[logging.Logger] = None,eval_metric_model = None,
                 early_stop=False, test_prob=False, fold_type='SKF',weights=None,multi_column_tfidf=None,custom_metric=None):

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
        self.weights = weights
        self.tf_vec = tf_vec
        self.stat_fe = stat_fe
        self.multi_column_tfidf = multi_column_tfidf
        self.gpu = gpu
        self.numpy_data = numpy_data
        self.handle_date = handle_date
        self.ohe_fe = ohe_fe
        self.label_encode = label_encode
        self.ordinal_encoder = ordinal_encoder
        self.target_encode = target_encode
        self.custom_metric = custom_metric
        self.eval_metric_model = eval_metric_model
        self.logger = logger or self._setup_default_logger()

        if self.metric == "custom" and callable(self.custom_metric):
            self.metric_name = self.custom_metric.__name__
        else:
            self.metric_name = self.metric
        
        self._validate_input()
        self.checkTarget()
        self._display_initial_info()

        if self.handle_date: 
            print(Fore.YELLOW + f"\nAdding Date Features")
            
            if self.train_data is not None:
                    self.train_data = self.date(
                        df=self.train_data,
                    )

            if self.test_data is not None:
                    self.test_data = self.date(
                        df=self.test_data,
                    )

        
        if self.tf_vec: 
            self.text_column = tf_vec.get('text_column', '')
            self.max_features = tf_vec.get('max_features', 1000)
            self.n_components = tf_vec.get('n_components', 10)
            print(Fore.YELLOW + f"\nTf-IDF Processing For Col: {self.text_column}")
            if self.train_data is not None:
                    self.train_data = self.apply_tfidf_svd(
                        df=self.train_data,
                        text_column=self.text_column,
                        max_features=self.max_features,
                        n_components=self.n_components
                    )

            if self.test_data is not None:
                    self.test_data = self.apply_tfidf_svd(
                        df=self.test_data,
                        text_column=self.text_column,
                        max_features=self.max_features,
                        n_components=self.n_components
                    )
                    
        if self.stat_fe: 
            print(Fore.YELLOW + f"\nAdding Stats Features")
            self.txt_columns = stat_fe.get('txt_columns', txt_columns)

            if self.train_data is not None:
                    self.train_data = self.text_stat(
                        df=self.train_data,
                        txt_cols=self.txt_columns,
                    )

            if self.test_data is not None:
                    self.test_data = self.text_stat(
                        df=self.test_data,
                        txt_cols=self.txt_columns,
                    )
        
        if self.ohe_fe:
            
            print(Fore.YELLOW + f"\n---> Adding OHE Features\n")
            self.cat_c = ohe_fe.get('cat_c', [])
            if self.train_data is not None and self.test_data is not None:
                self.train_data, self.test_data = self.ohe_transform(
                    train=self.train_data,
                    test=self.test_data,
                    cat_cols=self.cat_c, 
                )

        if self.label_encode:
            print(Fore.YELLOW + f"\n---> Applying Label Encoder\n")
            self.cat_c = label_encode.get('cat_c', [])
            if self.train_data is not None and self.test_data is not None:
                self.train_data, self.test_data = self.label_encode_transform(
                    train=self.train_data,
                    test=self.test_data,
                    cat_cols=self.cat_c, 
                )

        if self.ordinal_encoder:
            print(Fore.YELLOW + f"\n---> Applying Ordinal Encoder\n")
            self.cat_c = ordinal_encoder.get('cat_c', [])
            if self.train_data is not None and self.test_data is not None:
                self.train_data, self.test_data = self.ordinal_encode_transform(
                    train=self.train_data,
                    test=self.test_data,
                    cat_cols=self.cat_c, 
                )

        if self.target_encode:
            print(Fore.YELLOW + f"\n---> Applying Target Encoder\n")
            self.cat_c = target_encode.get('cat_c', [])
            self.target_col = target_encode.get('target_col',[])
            if self.train_data is not None and self.test_data is not None:
                self.train_data, self.test_data = self.factorize_and_encode(
                    train=self.train_data,
                    test=self.test_data,
                    cat_cols=self.cat_c, 
                    target_col=self.target_col, 
                )
                    
        if self.multi_column_tfidf:

            self.text_columns = multi_column_tfidf.get('text_columns', [])
            self.max_features = multi_column_tfidf.get('max_features', 1000)

            print(Fore.YELLOW + f"\nMulti-TF_IDF Processing For Columns: {self.text_columns}")

            if self.train_data is not None and self.test_data is not None:
                self.train_data, self.test_data = self.tf_fe(
                    train=self.train_data,
                    test=self.test_data,
                    text_columns=self.text_columns, 
                    max_features=self.max_features,
                )

        self.X_train = self.train_data.drop(self.target_column, axis=1).to_numpy() if self.numpy_data else self.train_data.drop(self.target_column, axis=1)
        self.y_train = self.train_data[self.target_column].to_numpy() if self.numpy_data else self.train_data[self.target_column]
        self.y_train = self.y_train.reshape(-1, 1) if self.model_name == 'TABNET' else self.y_train
        
        if self.test_data is not None:
            self.X_test = self.test_data.to_numpy() if self.numpy_data else self.test_data
        else:
            self.X_test = None

    @staticmethod
    def label_encode_transform(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list):
        label_encoders = {}
        
        for col in cat_cols:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])
            test[col] = test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            label_encoders[col] = le
        
        return train, test

    @staticmethod
    def ordinal_encode_transform(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list):
        encoders = {}
        
        for col in cat_cols:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            train[[col]] = encoder.fit_transform(train[[col]])
            test[[col]] = encoder.transform(test[[col]])
            encoders[col] = encoder
        
        return train, test

        
    @staticmethod
    def factorize_and_encode(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list, target_col) -> pd.DataFrame:
    
        combined = pd.concat([train, test], axis=0, ignore_index=True)
        
        for c in cat_cols:
            if c in combined.columns:
                combined[c], _ = combined[c].factorize()
                combined[c] -= combined[c].min()
                combined[c] = combined[c].astype("int32")  
                combined[c] = combined[c].astype("category") 
        
        for c in combined.columns:
            if c not in cat_cols:
                if combined[c].dtype == "float64":
                    combined[c] = combined[c].astype("float32")
                elif combined[c].dtype == "int64":
                    combined[c] = combined[c].astype("int32")
        
        train_encoded = combined.iloc[:len(train)].copy()
        test_encoded = combined.iloc[len(train):].reset_index(drop=True).copy()
    
        test_encoded = test_encoded.drop(columns=[target_col])
        
        return train_encoded, test_encoded
        
    @staticmethod
    def date(df): 
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['day_of_week'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7
        
        df.drop('date', axis=1, inplace=True)
    
        df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
        df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)
        df['year_lag_1'] = df['year'].shift(1)
        df['year_diff'] = df['year'] - df['year_lag_1']
    
        return df
            
    @staticmethod
    def ohe_transform(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list):
    
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
        train_ohe = pd.DataFrame(ohe.fit_transform(train[cat_cols]), 
                                 columns=ohe.get_feature_names_out(cat_cols), 
                                 index=train.index)
        
        test_ohe = pd.DataFrame(ohe.transform(test[cat_cols]), 
                                columns=ohe.get_feature_names_out(cat_cols), 
                                index=test.index)
    
        train = train.drop(columns=cat_cols).reset_index(drop=True)
        test = test.drop(columns=cat_cols).reset_index(drop=True)
    
        train = pd.concat([train, train_ohe.reset_index(drop=True)], axis=1)
        test = pd.concat([test, test_ohe.reset_index(drop=True)], axis=1)
    
        return train, test

    def CIBMTR_score(self,solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
        
        del solution[row_id_column_name]
        del submission[row_id_column_name]
        
        event_label = 'efs'
        interval_label = 'efs_time'
        prediction_label = 'prediction'
        for col in submission.columns:
            if not pandas.api.types.is_numeric_dtype(submission[col]):
                raise ParticipantVisibleError(f'Submission column {col} must be a number')
        merged_df = pd.concat([solution, submission], axis=1)
        merged_df.reset_index(inplace=True)
        merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
        metric_list = []
        for race in merged_df_race_dict.keys():
            indices = sorted(merged_df_race_dict[race])
            merged_df_race = merged_df.iloc[indices]
            c_index_race = concordance_index(
                            merged_df_race[interval_label],
                            -merged_df_race[prediction_label],
                            merged_df_race[event_label])
            metric_list.append(c_index_race)
        return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))
    
    @staticmethod
    def text_stat(df, txt_cols):
        stop_words = set(stopwords.words('english'))
        for col in tqdm(txt_cols, desc="Processing text columns"):
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame.")
                continue

            df[col] = df[col].fillna("")
            df[f'{col}_length'] = df[col].str.len()
            df[f'{col}_word_count'] = df[col].str.split().str.len()
            df[f'{col}_char_count'] = df[col].apply(lambda x: sum(len(word) for word in x.split()))
            df[f'{col}_avg_word_length'] = df[f'{col}_char_count'] / df[f'{col}_word_count'].replace(0, 1)

            df[f'{col}_punctuation_count'] = df[col].apply(lambda x: sum(1 for char in x if char in string.punctuation))
            df[f'{col}_capitalized_count'] = df[col].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
            df[f'{col}_special_char_count'] = df[col].apply(lambda x: sum(1 for char in x if not char.isalnum() and not char.isspace()))
            df[f'{col}_stopwords_count'] = df[col].apply(lambda x: sum(1 for word in x.split() if word.lower() in stop_words))

            df[f'{col}_unique_word_count'] = df[col].apply(lambda x: len(set(x.split())))
            df[f'{col}_lexical_diversity'] = df[f'{col}_unique_word_count'] / df[f'{col}_word_count'].replace(0, 1)

        return df

    @staticmethod              
    def apply_tfidf_svd(df, text_column, max_features=1000, n_components=10):
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            vectors = vectorizer.fit_transform(df[text_column])
            svd = TruncatedSVD(n_components)
            x_sv = svd.fit_transform(vectors)
            tfidf_df = pd.DataFrame(x_sv)
            cols = [(text_column + "_tfidf_" + str(f)) for f in tfidf_df.columns.to_list()]
            tfidf_df.columns = cols
            df = df.reset_index(drop=True)
            df = pd.concat([df, tfidf_df], axis="columns")
            df.drop(text_column, axis=1, inplace=True)
            return df
    @staticmethod       
    def tf_fe(train, test, text_columns, max_features=3000, analyzer='char_wb'):

        train_features = []
        test_features = []

        for col in tqdm(text_columns, desc="Processing text columns", unit="col"):
            train[col] = train[col].fillna("")
            test[col] = test[col].fillna("")  
            vectorizer = TfidfVectorizer(analyzer=analyzer, max_features=max_features)
            train_tfidf_col = vectorizer.fit_transform(train[col])
            test_tfidf_col = vectorizer.transform(test[col])
            train_tfidf_col = pd.DataFrame(train_tfidf_col.toarray(), columns=[f"tfidf_{col}_{i}" for i in range(train_tfidf_col.shape[1])])
            test_tfidf_col = pd.DataFrame(test_tfidf_col.toarray(), columns=[f"tfidf_{col}_{i}" for i in range(test_tfidf_col.shape[1])])
            train_features.append(train_tfidf_col)
            test_features.append(test_tfidf_col)

        train_with_tfidf = pd.concat([train, *train_features], axis=1)
        test_with_tfidf = pd.concat([test, *test_features], axis=1)

        return train_with_tfidf, test_with_tfidf

    def target_distribution(self, column_name, dataframe):
        value_counts = dataframe[column_name].value_counts()
    
        palette = [
            "#87CEEB",  # Sky Blue
            "#FFDAB9",  # Peachy Tone
            "#F0E68C",  # Khaki
            "#4682B4",  # Steel Blue
            "#FFA07A",  # Light Salmon
            "#7FFFD4",  # Aquamarine
            "#DDA0DD",  # Plum
            "#FFDEAD",  # Navajo White
            "#87CEFA",  # Light Sky Blue
            "#F08080"   # Light Coral
        ]
    
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.2, marker_colors=palette,
                             domain={'x': [0, 0.5], 'y': [0, 1]}, pull=[0.1, 0.1])) 
        fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values,
                             marker_color=palette, 
                             xaxis='x2', yaxis='y2'))
        fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                          marker=dict(line=dict(color='#000000', width=2)), selector=dict(type='pie'))
        fig.update_layout(
            title=f'Distribution of {column_name}',
            title_font_size=20,
            annotations=[
                {'text': '', 'x': 0.2, 'y': 0.5, 'showarrow': False, 'font': {'size': 16}}, 
                {'text': '', 'x': 0.8, 'y': 0.5, 'showarrow': False, 'font': {'size': 16}}
            ],
            grid={'rows': 1, 'columns': 2},
            xaxis=dict(domain=[0, 0.5]),
            xaxis2=dict(domain=[0.55, 1]),
            yaxis=dict(domain=[0, 1]),
            yaxis2=dict(domain=[0, 1], anchor='x2')
        )
    
        fig.show()

    def checkTarget(self):
        if self.train_data[self.target_column].dtype == 'object':
            raise ValueError('Encode Target First')
        
    def _display_initial_info(self):
        print(Fore.RED + f"*** AbdBase {self.current_version} ***\n")
        print(Fore.RED + " *** Available Settings *** \n")
        print(Fore.RED + "Available Models:", ", ".join([Fore.CYAN + model for model in self.model_name]))
        print(Fore.RED + "Available Metrics:", ", ".join([Fore.CYAN + metric for metric in self.metrics]))
        print(Fore.RED + "Available Problem Types:", ", ".join([Fore.CYAN + problem for problem in self.problem_types]))
        print(Fore.RED + "Available Fold Types:", ", ".join([Fore.CYAN + fold for fold in self.cv_types]))

        print(Fore.RED + "\n *** Configuration *** \n")
        print(Fore.RED + f"Problem Type Selected: {Fore.CYAN + self.problem_type.upper()}")
        print(Fore.RED + f"Metric Selected: {Fore.CYAN + self.metric.upper()}")
        print(Fore.RED + f"Fold Type Selected: {Fore.CYAN + self.fold_type}")
        print(Fore.RED + f"Calculate Train Probabilities: {Fore.CYAN + str(self.prob)}")  
        print(Fore.RED + f"Calculate Test Probabilities: {Fore.CYAN + str(self.test_prob)}")  
        print(Fore.RED + f"Early Stopping: {Fore.CYAN + str(self.early_stop)}")  
        print(Fore.RED + f"GPU: {Fore.CYAN + str(self.gpu)}")
        print(Fore.RED + f"Eval_Metric Selected is: {Fore.CYAN + str(self.eval_metric_model)}")


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
    
    def weighted_mean_absolute_error(self, y_true, y_pred, weights):
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    def rmsLe(self, y_true, y_pred):
        y_pred = np.maximum(y_pred, 1e-6)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def mape(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)

    def get_metric(self, y_true, y_pred, weights=None):
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
        elif self.metric == 'wmae' and weights is not None:
            return self.weighted_mean_absolute_error(y_true, y_pred, weights)
        elif self.metric == 'rmsle':
            return self.rmsLe(y_true, y_pred)
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred, squared=True)
        elif self.metric == "mape":
            return self.mape(y_true, y_pred)
        elif self.metric == 'custom' and callable(self.custom_metric):
            return self.custom_metric(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'")

    def Train_ML(self, params, model_name, e_stop=50,estimator=None,g_col=None,tab_net_train_params=None,optuna=False, V_weights=None,y_log=False):
        print(f"The EarlyStopping is {e_stop}") if optuna == False else None
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
        elif self.fold_type == 'RKF':
            kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats=1, random_state=self.seed)
        else:
            raise NotImplementedError("Select the Given Cv Statergy")

        train_scores = []
        oof_scores = []
        all_models = []
        oof_predictions = np.zeros((len(self.y_train), self.num_classes)) if self.num_classes > 2 else np.zeros(len(self.y_train))
        test_preds = (
            None if self.X_test is None else
            np.zeros((len(self.X_test), self.n_splits, self.num_classes)) if self.num_classes > 2 else
            np.zeros((len(self.X_test), self.n_splits))
        )
        
        cat_features_indices = [self.X_train.columns.get_loc(col) for col in self.cat_features] if model_name == 'CAT' else None
        
        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(self.X_train, self.y_train) if self.fold_type != 'GKF' else kfold.split(self.X_train, self.y_train, groups = self.X_train[g_col])
                                                         , desc="Training Folds", total=self.n_splits)):
            if self.numpy_data:
                X_train, X_val = self.X_train[train_idx], self.X_train[val_idx]
                y_train, y_val = self.y_train[train_idx], self.y_train[val_idx]
            else:
                X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_train, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            if y_log:
                y_train = np.log1p(y_train)
                y_val = np.log1p(y_val)
            
            # Sample The Test Weights
            def distribute_test_weights(test_sample_size, weights):
                repeated_weights = np.tile(weights, int(np.ceil(test_sample_size / len(weights))))[:test_sample_size]
                return repeated_weights
            
            if self.weights is not None:
#                 train_weights, val_weights = self.weights.iloc[train_idx], self.weights.iloc[val_idx]
                val_weights = distribute_test_weights(len(y_val), self.weights) # If Test Weights are Less || Sample Thm
                train_weights = np.ones(len(y_train)) # If Train Weights are None 

            callbacks = [early_stopping(stopping_rounds=e_stop, verbose=False)] if self.early_stop else None

            device = 'gpu' if self.gpu else 'cpu'
            xdevice = 'gpu_hist' if self.gpu else 'hist'
            cdevice = 'GPU' if self.gpu else 'CPU'
        
            if model_name == 'LGBM':
                model = lgb.LGBMClassifier(**params, random_state=self.seed, verbose=-1,device=device) if self.problem_type == 'classification' else lgb.LGBMRegressor(**params, random_state=self.seed, verbose=-1,
                device=device)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=self.eval_metric_model, callbacks=callbacks)

            elif model_name == 'TABNET':
                model = TabNetClassifier(**params, seed=self.seed, verbose=-1,device_name=device) if self.problem_type == 'classification' else TabNetRegressor(**params, seed=self.seed, verbose=-1,
                device_name=device)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=self.eval_metric_model, **tab_net_train_params)

            elif model_name == 'XGB':
                model = XGBClassifier(**params, random_state=self.seed, verbose=-1,tree_method=xdevice) if self.problem_type == 'classification' else XGBRegressor(**params, random_state=self.seed, verbose=-1,
                tree_method=xdevice)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=self.eval_metric_model, early_stopping_rounds=e_stop if self.early_stop else None, verbose=False)

            elif model_name == 'CAT':
                train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features_indices)
                val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features_indices)
                model = CatBoostClassifier(**params, random_state=self.seed, verbose=0,task_type=cdevice) if self.problem_type == 'classification' else CatBoostRegressor(**params, random_state=self.seed, verbose=0,
                task_type=cdevice)
                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=e_stop if self.early_stop else None)

            elif model_name == 'Voting':
                model = VotingClassifier(estimators=estimator,weights=V_weights if V_weights is not None else None) if self.problem_type == 'classification' else VotingRegressor(estimators=estimator,weights=V_weights if V_weights is not None else None)
                model.fit(X_train, y_train)
            
            elif model_name == 'Ridge':
                model = Ridge(**params)
                model.fit(X_train, y_train)
                
            elif model_name == 'LR':
                model = LinearRegression()
                model.fit(X_train, y_train)
            else:
                raise ValueError("model_name must be 'LGBM' or 'CAT'.")

            if self.problem_type == 'classification':
                y_train_pred = model.predict_proba(X_train)[:, 1] if self.num_classes == 2 else model.predict_proba(X_train) 
                y_val_pred = model.predict_proba(X_val)[:, 1] if self.num_classes == 2 else model.predict_proba(X_val) 
            else:
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
            if y_log:
                y_train_pred = np.expm1(y_train_pred)
                y_val_pred = np.expm1(y_val_pred)
                y_train = np.expm1(y_train)
                y_val = np.expm1(y_val)
                
            oof_predictions[val_idx] = y_val_pred

            if self.num_classes == 2:
                y_train_pred = np.round(y_train_pred)
                y_val_pred = np.round(y_val_pred)

            elif self.num_classes > 2:
                y_train_pred = np.argmax(y_train_pred, axis =1)
                y_val_pred = np.argmax(y_val_pred, axis =1)

            if self.metric == "accuracy":
                train_scores.append(accuracy_score(y_train, y_train_pred))
                oof_scores.append(accuracy_score(y_val, y_val_pred))
            elif self.metric == "roc_auc":
                train_scores.append(roc_auc_score(y_train, y_train_pred, multi_class="ovr" if self.num_classes > 2 else None))
                oof_scores.append(roc_auc_score(y_val, y_val_pred, multi_class="ovr" if self.num_classes > 2 else None))
                
            elif self.metric == 'wmae' and self.weights is not None:
                train_scores.append(self.get_metric(y_train, y_train_pred, train_weights))
                oof_scores.append(self.get_metric(y_val, y_val_pred, val_weights))

            else:
                train_scores.append(self.get_metric(y_train, y_train_pred))
                oof_scores.append(self.get_metric(y_val, y_val_pred))
    
            if self.X_test is not None:
                if self.problem_type == 'classification':
                    test_preds[:, fold] = model.predict_proba(self.X_test)[:, 1] if self.num_classes == 2 else model.predict_proba(self.X_test)
                elif model_name == 'TABNET':
                    pred = model.predict(self.X_test)
                    test_preds[:, fold] = pred.squeeze() 
                else:
                    test_preds[:, fold] = model.predict(self.X_test)

            print(f"Fold {fold + 1} - Train {self.metric_name.upper()}: {train_scores[-1]:.4f}, OOF {self.metric_name.upper()}: {oof_scores[-1]:.4f}") if optuna == False else None
            all_models.append(model)
            clear_output(wait=True) if optuna == False else None
            
        mean_train_scores = f"{np.mean(train_scores):.4f}"
        mean_off_scores = f"{np.mean(oof_scores):.4f}"
        
        print(f"Overall Train {self.metric_name.upper()}: {mean_train_scores}") if optuna == False else None
        print(f"Overall OOF {self.metric_name.upper()}: {mean_off_scores} ") if optuna == False else None

        mean_test_preds = test_preds.mean(axis=1) if self.X_test is not None else None

        if y_log:
            mean_test_preds = np.expm1(mean_test_preds)

        if self.prob:
            oof_predictions = oof_predictions
        elif not self.prob and self.num_classes == 2:
            oof_predictions = np.round(oof_predictions)
        elif not self.prob and self.num_classes > 2:
            oof_predictions = np.argmax(oof_predictions, axis=1)

        if self.test_data is not None:

            if self.test_prob:
                mean_test_preds = mean_test_preds
            elif not self.test_prob and self.num_classes == 2:
                mean_test_preds = np.round(mean_test_preds)
            elif not self.test_prob and self.num_classes > 2:
                mean_test_preds = np.argmax(mean_test_preds, axis=1)

        return oof_predictions, mean_test_preds , model , all_models , mean_off_scores , mean_train_scores
    
    def _setup_default_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
            
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def RUN_OPTUNA(
            self, MODEL_NAME: str, PARAMS: Dict[str, Any], DIRECTION: str = 'minimize', 
            TRIALS: int = 10, SEED: int = 42, ENABLE_PRUNER: bool = False, 
            PRUNER_PARAMS: Optional[Dict[str, Any]] = None, y_log: bool = False) -> Dict[str, Any]:
        
        sampler = optuna.samplers.TPESampler(seed=SEED)
        
        pruner = None
        if ENABLE_PRUNER:
            pruner_config = PRUNER_PARAMS or {
                'n_startup_trials': 5,
                'n_warmup_steps': 3,
                'n_valid_steps': 3
            }
            pruner = optuna.pruners.MedianPruner(**pruner_config)
        
        study = optuna.create_study(sampler=sampler, direction=DIRECTION, pruner=pruner)
        
        best_scores = {'train_score': None, 'val_score': None}
        
        def objective(trial):
            train_score, val_score = self.OPTUNE_TRAIN(trial, MODEL_NAME=MODEL_NAME, PARAMS=PARAMS, y_log=y_log)
            
            if best_scores['val_score'] is None or (
                (DIRECTION == 'minimize' and val_score < best_scores['val_score']) or
                (DIRECTION == 'maximize' and val_score > best_scores['val_score'])
            ):
                best_scores['train_score'] = train_score
                best_scores['val_score'] = val_score
            
            return val_score
        
        try:
            study.optimize(objective, n_trials=TRIALS)
            
            self.logger.info(Fore.RED + f"--> Best Train Score for {MODEL_NAME}: " + 
                            Fore.CYAN + f"{best_scores['train_score']:.4f}")
            self.logger.info(Fore.RED + f"--> Best Validation Score for {MODEL_NAME}: " + 
                            Fore.CYAN + f"{best_scores['val_score']:.4f}")
            self.logger.info(Fore.RED + f"--> Best Parameters: " + Fore.CYAN + f"{study.best_params}")
            
            return study
        
        except Exception as e:
            self.logger.error(f"Optuna Optimization Failed: {str(e)}")
            raise
    
    def OPTUNE_TRAIN(self, trial: optuna.trial.Trial, MODEL_NAME: str = "", optuna=True,
            PARAMS: Optional[Dict[str, Union[Tuple[Union[int, float], Union[int, float]], Any]]] = None,
            y_log: bool = False) -> Tuple[float, float]:
        
        params = PARAMS.copy() if PARAMS else {}
        
        for param, value in params.items():
            try:
                if isinstance(value, tuple) and len(value) == 2:
                    if isinstance(value[0], int):
                        params[param] = trial.suggest_int(param, value[0], value[1])
                    elif isinstance(value[0], float):
                        params[param] = trial.suggest_float(param, value[0], value[1], log=True)
            except Exception as e:
                self.logger.error(f"Error suggesting parameter {param}: {e}")
                raise
        
        try:
            result = self.Train_ML(params=params, model_name=MODEL_NAME, e_stop=40, estimator=None, 
                                   g_col=None, tab_net_train_params=None, optuna=optuna, y_log=y_log)
            
            test_score = result[4]
            train_score = result[5]
            try:
                test_score = float(test_score)
                train_score = float(train_score)
                return train_score, test_score
            except ValueError as e:
                raise ValueError(f"Score '{test_score}' and {train_score} is not a valid float. Original error: {e}")
        
        except Exception as e:
            self.logger.error(f"Training failed for {MODEL_NAME}: {e}")
            raise        
