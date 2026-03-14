import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.utils import load_object
from src.constants import ARTIFACTS_DIR, PREPROCESSOR_FILE_NAME, MODEL_FILE_NAME


@dataclass
class ChurnPredictionInput:
    """Maps to the raw feature columns expected by the model."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "gender": self.gender,
            "SeniorCitizen": self.SeniorCitizen,
            "Partner": self.Partner,
            "Dependents": self.Dependents,
            "tenure": self.tenure,
            "PhoneService": self.PhoneService,
            "MultipleLines": self.MultipleLines,
            "InternetService": self.InternetService,
            "OnlineSecurity": self.OnlineSecurity,
            "OnlineBackup": self.OnlineBackup,
            "DeviceProtection": self.DeviceProtection,
            "TechSupport": self.TechSupport,
            "StreamingTV": self.StreamingTV,
            "StreamingMovies": self.StreamingMovies,
            "Contract": self.Contract,
            "PaperlessBilling": self.PaperlessBilling,
            "PaymentMethod": self.PaymentMethod,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalCharges": self.TotalCharges,
        }])


class PredictionPipeline:
    def __init__(self):
        preprocessor_path = os.path.join(
            ARTIFACTS_DIR, "preprocessor", PREPROCESSOR_FILE_NAME
        )
        model_path = os.path.join(
            ARTIFACTS_DIR, "models", MODEL_FILE_NAME
        )
        self.preprocessor = load_object(preprocessor_path)
        self.model = load_object(model_path)
        logger.info("PredictionPipeline: preprocessor & model loaded.")

    def predict(self, input_data: ChurnPredictionInput) -> dict:
        """
        Accepts a ChurnPredictionInput, returns prediction label and probability.
        """
        try:
            df = input_data.to_dataframe()
            X = self.preprocessor.transform(df)
            prediction = int(self.model.predict(X)[0])
            proba = (
                float(self.model.predict_proba(X)[0][1])
                if hasattr(self.model, "predict_proba")
                else None
            )
            result = {
                "prediction": prediction,
                "churn": "Yes" if prediction == 1 else "No",
                "churn_probability": round(proba, 4) if proba is not None else "N/A",
            }
            logger.info(f"Prediction result: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)
