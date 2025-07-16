import sys
import os
# Add this snippet before importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path: str, obj: object) -> None:
    try:
        # Ensure the directory exists before saving
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        reports ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            grid = GridSearchCV(model,para,cv=3)
            grid.fit(X_train,y_train)
            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            reports[list(models.keys())[i]]=test_model_score

        return reports

    except Exception as e:
        raise CustomException(e,sys)