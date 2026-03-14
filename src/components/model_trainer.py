import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import CustomException
from src.entity import ModelTrainerConfig
from src.utils import evaluate_models, save_object


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
}

PARAMS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },
}

ACCURACY_THRESHOLD = 0.75


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        """
        Runs GridSearchCV for each model, picks the best one,
        saves it, and returns its name + accuracy.
        """
        logger.info(">>> Model Training started")
        try:
            report = evaluate_models(X_train, y_train, X_test, y_test, MODELS, PARAMS)

            best_name = max(report, key=lambda k: report[k]["score"])
            best_score = report[best_name]["score"]
            best_model = report[best_name]["model"]

            logger.info(f"Best model: {best_name}  |  Accuracy: {best_score:.4f}")

            if best_score < ACCURACY_THRESHOLD:
                raise ValueError(
                    f"No model crossed the accuracy threshold of {ACCURACY_THRESHOLD}. "
                    f"Best was {best_name} at {best_score:.4f}"
                )

            save_object(self.config.trained_model_file_path, best_model)
            logger.info(f"Model saved: {self.config.trained_model_file_path}")
            logger.info(">>> Model Training completed")

            return best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)
