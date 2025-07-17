import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline():
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            pred =model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 math_score:int,
                 reading_score:int,):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.math_score=math_score
        self.reading_score = reading_score
        
    
    def getdata_as_DataFrame(self):
        try:
            custom_data_input_dict={
              "gender":[self.gender],
              "race_ethnicity":[self.race_ethnicity],
              "parental_level_of_education":[self.parental_level_of_education],
              "lunch":[self.lunch],
              "test_preparation_course":[self.test_preparation_course],
              "math_score":[self.math_score],
              "reading_score":[self.reading_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)