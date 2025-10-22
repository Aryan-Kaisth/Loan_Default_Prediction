# src/components/model_trainer.py
import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model_trainer", "logistic.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        # Make sure directory exists
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Trains a Histogram Gradient Boosting Classifier model and evaluates it.
        Saves the trained model as a pickle file.
        Returns the accuracy and classification report.
        """
        try:
            logging.info("Model training started")
            # Initialize model
            log = LogisticRegression(C=0.035,penalty='l2',solver='lbfgs',class_weight={0: 1, 1: 45},max_iter=1000,random_state=42)

            # Train model
            log.fit(X_train, y_train)
            logging.info("Model training completed")

            # Predict on test set
            y_pred = log.predict(X_test)

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            logging.info(f"Model Accuracy: {accuracy}")
            logging.info(f"Precision Score:\n{precision}")
            logging.info(f"Recall Score:\n{recall}")
            logging.info(f"Classification Report:\n{report}")

            # Save model
            save_object(self.config.model_file_path, log)
            logging.info(f"Trained model saved at: {self.config.model_file_path}")

            return log

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
        

# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation

    # Paths to train and test data
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize the transformer
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    model = ModelTrainer()
    model.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)