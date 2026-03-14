import os
from src.constants import (
    ARTIFACTS_DIR,
    RAW_FILE_NAME,
    TRAIN_SPLIT_FILE,
    TEST_SPLIT_FILE,
    PREPROCESSOR_FILE_NAME,
    MODEL_FILE_NAME,
)
from src.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    def __init__(self):
        self.artifacts_dir = ARTIFACTS_DIR
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        raw_dir = os.path.join(self.artifacts_dir, "data_ingestion")
        os.makedirs(raw_dir, exist_ok=True)
        return DataIngestionConfig(
            raw_data_path=os.path.join(raw_dir, RAW_FILE_NAME),
            train_data_path=os.path.join(raw_dir, TRAIN_SPLIT_FILE),
            test_data_path=os.path.join(raw_dir, TEST_SPLIT_FILE),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        preprocessor_dir = os.path.join(self.artifacts_dir, "preprocessor")
        os.makedirs(preprocessor_dir, exist_ok=True)
        return DataTransformationConfig(
            preprocessor_obj_file_path=os.path.join(preprocessor_dir, PREPROCESSOR_FILE_NAME)
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_dir = os.path.join(self.artifacts_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        return ModelTrainerConfig(
            trained_model_file_path=os.path.join(model_dir, MODEL_FILE_NAME)
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        eval_dir = os.path.join(self.artifacts_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        return ModelEvaluationConfig(
            metric_file_path=os.path.join(eval_dir, "metrics.json")
        )
