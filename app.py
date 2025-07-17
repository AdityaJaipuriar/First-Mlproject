from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=request.form.get('math_score'),
            reading_score=request.form.get('reading_score'),
        )
        pred_df = data.getdata_as_DataFrame()
        print(pred_df)
        try:
            predict_pipeline=PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return "Something went wrong in prediction pipeline."
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)