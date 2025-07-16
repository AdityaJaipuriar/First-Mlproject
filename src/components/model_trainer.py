import os
import sys
# Add this snippet before importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

from sklearn.metrics import r2_score

@dataclass
class ModelConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,X_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            models = {
                "Linear Regression":LinearRegression(),
                "Logistic Regression":LogisticRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNeighbors":KNeighborsRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Support Vector Machine":SVR(),
                "Adaboost":AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Xgboost":XGBRegressor(),
                "catboost":CatBoostRegressor(verbose=False)
            }
            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # Best Model score
            best_model_score = max(sorted(model_report.values()))
            # Best Model Name
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]


            best_model = models[best_model_name]
            
            if (best_model_score<0.7):
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2score=r2_score(y_test,predicted)
            return r2score
        except Exception as e:
            raise CustomException(e,sys)