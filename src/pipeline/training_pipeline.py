import sys
from src.logger import logger
from src.exception import CustomException
from src.config import ConfigurationManager
from src.components import (
    DataIngestion,
    DataTransformation,
    ModelTrainer,
    ModelEvaluation,
)


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        logger.info("=" * 60)
        logger.info("          TRAINING PIPELINE STARTED")
        logger.info("=" * 60)
        try:
            # ── Stage 1 : Data Ingestion ──────────────────────────────
            logger.info("Stage 1 → Data Ingestion")
            ingestion = DataIngestion(self.config.get_data_ingestion_config())
            train_path, test_path = ingestion.initiate_data_ingestion()

            # ── Stage 2 : Data Transformation ────────────────────────
            logger.info("Stage 2 → Data Transformation")
            transformation = DataTransformation(
                self.config.get_data_transformation_config()
            )
            X_train, y_train, X_test, y_test, preprocessor_path = (
                transformation.initiate_data_transformation(train_path, test_path)
            )

            # ── Stage 3 : Model Training ──────────────────────────────
            logger.info("Stage 3 → Model Training")
            trainer = ModelTrainer(self.config.get_model_trainer_config())
            best_model_name, best_score = trainer.initiate_model_training(
                X_train, y_train, X_test, y_test
            )

            # ── Stage 4 : Model Evaluation ────────────────────────────
            logger.info("Stage 4 → Model Evaluation")
            evaluator = ModelEvaluation(self.config.get_model_evaluation_config())
            model_path = self.config.get_model_trainer_config().trained_model_file_path
            metrics = evaluator.initiate_model_evaluation(X_test, y_test, model_path)

            logger.info("=" * 60)
            logger.info(f"  Best Model  : {best_model_name}")
            logger.info(f"  Accuracy    : {metrics['accuracy']}")
            logger.info(f"  F1 Score    : {metrics['f1_score']}")
            logger.info(f"  ROC-AUC     : {metrics['roc_auc']}")
            logger.info("          TRAINING PIPELINE COMPLETED")
            logger.info("=" * 60)

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
