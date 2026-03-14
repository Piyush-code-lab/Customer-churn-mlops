from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logger

if __name__ == "__main__":
    logger.info("Launching Training Pipeline from main.py")
    pipeline = TrainingPipeline()
    metrics = pipeline.run()
    print("\n========== Final Evaluation Metrics ==========")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "classification_report"):
            print(f"  {k:20s}: {v}")
    print("==============================================\n")
