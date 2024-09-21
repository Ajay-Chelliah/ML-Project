import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import  save_object
from src.utils import  evaluate_model

@dataclass
class Model_Trainer_Config: # Creatning a class variable to store the ML model path
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class Model_Trainer:
    """
     1. Creating a object for the Model_Trainer_Config class to access the path variable
    """
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()
    def initiate_train(self,train_arr,test_arr):
        """
            1. Splitting the train and test data into input and target.
            2. Creating a models dictinory containing all the models we are using.
            3. Calling the evaluate_model function in the utils.py for training the model
            4. Storing the best_score and list of all the model names with their score from the result of above function.
            5. Calling the save_object function in the utils.py to save the model in the modelpath.
            6. Returning the r2_score of the best model
        """
        try:
            logging.info("Splitting training and test input data")
            X_train,Y_train,X_test,Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            models = {
                'Random Forest': RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier" : CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, models = models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)


