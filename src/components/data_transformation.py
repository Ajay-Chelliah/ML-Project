import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
        Like in the data ingestion file , we create a variable to store the file path where the preprocessor will get stored.
    """
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    """
        This class is to perform the transformation steps like scaling and encoding.
    """
    def __init__(self): # Creating a object of the above config class to access the path variable.
        self.DataTransformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
            1. Finding the numerical and categorical columns.
            2. Createing a Pipeline for transforming both numerical and categorical columns seperately.
            3. Creating a Column Transformer to run the steps in a correct order.
            4. Returns the Column Transformer variable (preprocessor).
        """
        try:
            num_cols = ['writing_score','reading_score']
            cat_cols = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("Scaler" ,StandardScaler())
                ]
            )
            logging.info("Numerical columns scalind completed")
            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ("Encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ]
            )
            logging.info("Column Transformer has been created")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
            1. Reading the train and test file.
            2. Getting the preprocessor from the above function.
            3. Finding the input and target columns for both train and test data.
            4. Fitting and transforming the input data using the preprocessor.
            5. Concatenating the input and target columns column-wise with the respective rows for both train and test data.
            6. Saving the preprocessor object.
            7. Returning the transformed train and test data and the preprocessor object.
            8. Logging all the important steps.
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read the train and test csv file")

            logging.info("Obtaining the preprocessor object")

            preprocessing_obj = self.get_data_transformer_obj()
            target_col = 'math_score'
            num_col = ['writing_score','reading_score']

            input_train_features = train_df.drop(columns=[target_col],axis=1)
            target_train_features = train_df[target_col]
            
            input_test_features = test_df.drop(columns=[target_col],axis=1)
            target_test_features = test_df[target_col]
            
            input_train_features_arr = preprocessing_obj.fit_transform(input_train_features)
            input_test_features_arr = preprocessing_obj.fit_transform(input_test_features)

            train_arr = np.c_[
                input_train_features_arr , np.array(target_train_features)    
            ]
            test_arr = np.c_[input_test_features_arr,np.array(target_test_features)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.DataTransformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            """
                This is a function defined in the util.py file 
                which create a mentioned directory and store the preprocessor object using dill module.
            """

            return(
                train_arr,
                test_arr,
                self.DataTransformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)