import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_numpy_array_data, save_object, read_csv_file, read_yaml_file
from src.components.data_ingestion import DataIngestion, DataIngestionConfig

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    transformed_train_file_path: str = os.path.join("artifacts", "data_transformation", "train.npy")
    transformed_test_file_path: str = os.path.join("artifacts", "data_transformation", "test.npy")

class DataTransformation:
    SCHEMA_PATH = os.path.join("config", "schema.yaml")

    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            # Read schema using your utility function
            self.schema = read_yaml_file(self.SCHEMA_PATH)
            self.drop_col = self.schema.get("drop_col", [])
            self.num_cols = self.schema.get("numeric", [])
            self.ord_cols = self.schema.get("ordinal", [])
            self.nom_cols = self.schema.get("nominal", [])
            self.target_column = self.schema.get("target")
            logging.info(
                f"Schema loaded successfully. Drop: {self.drop_col}, "
                f"Numerical: {self.num_cols}, Ordinal: {self.ord_cols}, "
                f"Nominal: {self.nom_cols}, Target: {self.target_column}"
            )
        except Exception as e:
            logging.error("Error initializing DataTransformation with schema")
            raise CustomException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Debt Burden: LoanAmount relative to CreditScore
            df['LoanPerCreditScore'] = df['LoanAmount'] / (df['CreditScore'] + 1)  # avoid divide by zero

            # Income per Credit Line
            df['IncomePerCreditLine'] = df['Income'] / (df['NumCreditLines'] + 1)

            # Interest burden relative to Income
            df['InterestOverIncome'] = df['InterestRate'] / (df['Income'])

            # Monthly Loan Payment Estimate (simplified)
            df['MonthlyPayment'] = df['LoanAmount'] / df['LoanTerm']

            # Credit Score to Age ratio (stability measure)
            df['CreditScorePerAge'] = df['CreditScore'] / (df['Age'])

            # Employment stability ratio (MonthsEmployed / Age)
            df['EmploymentStability'] = df['MonthsEmployed'] / (df['Age'])

            # Debt-to-Loan ratio
            df['DTI_LoanRatio'] = df['DTIRatio'] / (df['LoanAmount'])

            # Income to Loan term ratio
            df['IncomePerLoanTerm'] = df['Income'] / (df['LoanTerm'])

            # Loan-to-Income Ratio (LTI)
            df['LTI'] = df['LoanAmount'] / df['Income']

            # Years Employed (from MonthsEmployed)
            df['YearsEmployed'] = df['MonthsEmployed'] / 12

            # Employment to Loan Term Ratio
            df['EmploymentToLoanTerm'] = df['MonthsEmployed'] / df['LoanTerm']

            df.drop(columns=self.drop_col, inplace=True, axis=1)
            return df
        except Exception as e:
            logging.error("Error in feature engineering")
            raise CustomException(e, sys)

    def get_preprocessor_pipeline(self):
        try:
            education_order = ["High School", "Bachelor's", "Master's", "PhD"]
            preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.num_cols),
            ('ord', OrdinalEncoder(categories=[education_order]), self.ord_cols),
            ('nom', OneHotEncoder(drop='first', sparse_output=True), self.nom_cols)], remainder="passthrough")
            return preprocessor
        except Exception as e:
            logging.error("Error creating preprocessor Pipeline")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train and test data")
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            logging.info("Applying feature engineering on train and test data")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            X_train = train_df.drop(columns=[self.target_column], axis=1)
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column], axis=1)
            y_test = test_df[self.target_column]

            logging.info("Creating preprocessing Pipeline")
            preprocessor = self.get_preprocessor_pipeline()

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Saving transformed data and preprocessor")
            save_numpy_array_data(self.config.transformed_train_file_path, np.c_[X_train_transformed, y_train])
            save_numpy_array_data(self.config.transformed_test_file_path, np.c_[X_test_transformed, y_test])
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            
            logging.info(f"Data transformation completed and saved successfully at {self.config.preprocessor_obj_file_path}")
            return X_train_transformed, X_test_transformed, y_train, y_test
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)
        

# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig

    # Paths to train and test data
    config = DataIngestionConfig()
    obj = DataIngestion(config=config)
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Initialize the transformer
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    # Quick checks
    print("Transformed X_train shape:", X_train_transformed.shape)
    print("Transformed X_test shape:", X_test_transformed.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)