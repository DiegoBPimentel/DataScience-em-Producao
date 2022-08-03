from flask import Flask, request, Response 
import pandas as pd
from rossmann.Rossmann import Rossmann 
import pickle 



#load model
model = pickle.load(open('/home/diego/repos/DataScience-em-Producao/model/model_rosmann.pkl','rb')) 

#initialize API
app = Flask ( __name__) #instanciando a classe 

@app.route('/rossmann/predict' , methods = ['POST']) 

def rossmann_predict(): 
    test_json = request.get_json()
        
    if test_json: 
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])  
        else: 
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        #Instantiate Rossmann Class
        pipeline = Rossmann() 
        
        #data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        #feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        #data preparation
        df3 = pipeline.data_preparation(df2)
        
        #preidction 
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response 
        
            
    else:
        return Response ('{}', status=200, mimetype ='application/json' )
        
if __name__ == '__main__': 
    app.run ('0.0.0.0')   
