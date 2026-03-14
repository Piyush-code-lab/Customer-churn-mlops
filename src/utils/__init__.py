import os
import dill
import json
import numpy as np
from src.exception import CustomException
from src.logger import logger
import sys


def save_object(file_path: str, obj) -> None:
    """Serialize and save any Python object using dill."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a serialized Python object."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path: str, data: dict) -> None:
    """Save a dictionary as a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """
    Train each model with GridSearchCV and return test scores.
    Returns a dict {model_name: test_score}.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    report = {}
    try:
        for name, model in models.items():
            param_grid = params.get(name, {})
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logger.info(f"[{name}] Best params: {gs.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = {"score": score, "model": best_model}
            logger.info(f"[{name}] Accuracy: {score:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
