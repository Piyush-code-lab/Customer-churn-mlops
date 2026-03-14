import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.logger import logger
from src.exception import CustomException
from src.entity import DataTransformationConfig
from src.constants import TARGET_COLUMN, DROP_COLUMNS
from src.utils import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Drop irrelevant columns and fix known dtype issues."""
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")
        # TotalCharges has stray spaces that prevent numeric conversion
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        return df

    @staticmethod
    def _encode_target(series: pd.Series) -> np.ndarray:
        """Yes → 1, No → 0"""
        return (series.str.strip().str.lower() == "yes").astype(int).values

    def _build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Construct the ColumnTransformer based on column dtypes."""
        df = df.drop(columns=[TARGET_COLUMN], errors="ignore")

        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        logger.info(f"Numerical columns  : {numerical_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ])
        return preprocessor

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train/test CSVs → cleans → fits preprocessor on train →
        transforms both → saves preprocessor object.
        Returns (X_train, y_train, X_test, y_test, preprocessor_path).
        """
        logger.info(">>> Data Transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self._clean_dataframe(train_df)
            test_df = self._clean_dataframe(test_df)

            y_train = self._encode_target(train_df[TARGET_COLUMN])
            y_test = self._encode_target(test_df[TARGET_COLUMN])

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            X_test = test_df.drop(columns=[TARGET_COLUMN])

            preprocessor = self._build_preprocessor(train_df)
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logger.info(f"Preprocessor saved: {self.config.preprocessor_obj_file_path}")
            logger.info(">>> Data Transformation completed")

            return X_train_arr, y_train, X_test_arr, y_test, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
