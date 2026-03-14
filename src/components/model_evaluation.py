import sys
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.logger import logger
from src.exception import CustomException
from src.entity import ModelEvaluationConfig
from src.utils import load_object, save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(
        self, X_test, y_test, model_path: str
    ) -> dict:
        """
        Loads saved model → evaluates on test set →
        saves metrics JSON → returns metrics dict.
        """
        logger.info(">>> Model Evaluation started")
        try:
            model = load_object(model_path)
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else "N/A",
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
            }

            save_json(self.config.metric_file_path, metrics)

            logger.info(f"Accuracy  : {metrics['accuracy']}")
            logger.info(f"Precision : {metrics['precision']}")
            logger.info(f"Recall    : {metrics['recall']}")
            logger.info(f"F1 Score  : {metrics['f1_score']}")
            logger.info(f"ROC-AUC   : {metrics['roc_auc']}")
            logger.info(f"Metrics saved at: {self.config.metric_file_path}")
            logger.info(">>> Model Evaluation completed")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)
