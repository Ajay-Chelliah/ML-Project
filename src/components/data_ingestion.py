import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import Model_Trainer_Config
from src.components.model_trainer import Model_Trainer

@dataclass  # This abstract method helps us to create class variables without a __init__ function
class DataIngestionConfig:
    '''
     This class is used to create class variables to store the path name of the data files 
     which includes directory name and file name.
    '''
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    """
    This class is to perform the data ingestion steps
    """
    def __init__(self): # Creates an object for the above config class to access the file paths.
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self): 
        """
            1. Reading the raw data
            2. Creating the directory mentioned in the config class if not exist
            3. Saving the raw data as raw.csv inside the artifact directory
            4. Splitting the raw data into train and test data and saving them as train.csv and test.csv in artifacts folder.
            5. Returning the train and test data path so that those can be accessed by the other py files like data_transformation.
            6. Logging is included in all the important steps to keep track of the steps executed to find and solve any error easily.
        """
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        
            logging.info("Train test split initiated")
            
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)        
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)        
            
            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e: # Raise a custom exception defined in the exception.py file
            raise CustomException(e,sys)

if __name__ == "__main__":
    """
        Creates an object for the main class DataIngestion 
        Calls the important function inside it to initiate the data ingestion process
        Creates an object for the Data Transformation class
        Call the function inside the above class to transform the data (numerical and categorical)
    """
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr , preprocessor= data_transformation.initiate_data_transformation(train_data,test_data)
   
    model_trainer = Model_Trainer()
    print(model_trainer.initiate_train(train_arr,test_arr))
