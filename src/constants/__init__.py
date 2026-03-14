import os

# Paths
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Raw data
TRAIN_FILE_NAME = "train.csv"
RAW_FILE_NAME = "raw.csv"
TRAIN_SPLIT_FILE = "train_split.csv"
TEST_SPLIT_FILE = "test_split.csv"

# Target column
TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]

# Model & preprocessor artifact names
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
MODEL_FILE_NAME = "model.pkl"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparameter tuning
CV_FOLDS = 3
