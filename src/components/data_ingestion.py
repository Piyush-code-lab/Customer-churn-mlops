import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.entity import DataIngestionConfig
from src.constants import DATA_DIR, TRAIN_FILE_NAME, TEST_SIZE, RANDOM_STATE


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """
        Reads raw data → saves a copy → splits into train/test.
        Returns paths to train and test CSVs.
        """
        logger.info(">>> Data Ingestion started")
        try:
            source_path = os.path.join(DATA_DIR, TRAIN_FILE_NAME)
            df = pd.read_csv(source_path)
            logger.info(f"Dataset loaded: {df.shape}")

            # Save raw copy
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            # Train / test split
            train_df, test_df = train_test_split(
                df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Churn"]
            )
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(
                f"Train shape: {train_df.shape} | Test shape: {test_df.shape}"
            )
            logger.info(">>> Data Ingestion completed")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
