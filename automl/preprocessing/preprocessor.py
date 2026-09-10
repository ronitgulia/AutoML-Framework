"""
AutoML Framework - Preprocessing Module
Handles imputation, encoding, scaling and outlier removal.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    impute_strategy : str
        Strategy for numerical imputation. One of {'mean', 'median', 'knn'}.
    scale : bool
        Whether to apply feature scaling.
    scaler_type : str
        Type of scaler. One of {'standard', 'minmax', 'robust'}.
    cat_encoding : str
        Categorical encoding strategy. One of {'onehot', 'ordinal'}.
    remove_outliers : bool
        Whether to clip outliers using IQR.
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        scale: bool = True,
        scaler_type: str = "standard",
        cat_encoding: str = "onehot",
        remove_outliers: bool = False,
    ):
        self.impute_strategy = impute_strategy
        self.scale = scale
        self.scaler_type = scaler_type
        self.cat_encoding = cat_encoding
        self.remove_outliers = remove_outliers

    def fit(self, X: pd.DataFrame, y=None):
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Numerical imputer
        if self.impute_strategy == "knn":
            self.num_imputer_ = KNNImputer(n_neighbors=5)
        else:
            self.num_imputer_ = SimpleImputer(strategy=self.impute_strategy)
        self.num_imputer_.fit(X[self.num_cols_])

        # Outlier bounds (IQR)
        if self.remove_outliers and self.num_cols_:
            q1 = X[self.num_cols_].quantile(0.25)
            q3 = X[self.num_cols_].quantile(0.75)
            iqr = q3 - q1
            self.lower_ = q1 - 1.5 * iqr
            self.upper_ = q3 + 1.5 * iqr

        # Scaler
        if self.scale and self.num_cols_:
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }
            self.scaler_ = scalers.get(self.scaler_type, StandardScaler())
            self.scaler_.fit(X[self.num_cols_])

        # Categorical encoder
        if self.cat_cols_:
            if self.cat_encoding == "onehot":
                self.cat_encoder_ = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
            else:
                self.cat_encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            self.cat_encoder_.fit(X[self.cat_cols_])

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()

        # Impute numerics
        X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])

        # Clip outliers
        if self.remove_outliers and self.num_cols_:
            X[self.num_cols_] = X[self.num_cols_].clip(
                lower=self.lower_, upper=self.upper_, axis=1
            )

        # Scale numerics
        if self.scale and self.num_cols_:
            X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])

        # Encode categoricals
        if self.cat_cols_:
            cat_encoded = self.cat_encoder_.transform(X[self.cat_cols_])
            num_part = X[self.num_cols_].values
            return np.hstack([num_part, cat_encoded])

        return X[self.num_cols_].values
