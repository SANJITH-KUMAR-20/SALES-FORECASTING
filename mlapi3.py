import fastapi as ft
from fastapi import FastAPI
import pickle 
import pandas as pd
from pydantic import BaseModel
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class DropColumn(BaseEstimator,TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X.drop(self.columns,axis = 1) if self.columns else X.drop(X.columns,axis = 1)
    
class ToTypeStr(BaseEstimator,TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns
    def fit(self,X,y = None):
        return self
    def transform(self,X,y=None):
        columns_to_trans = X.columns
        if self.columns:
            columns_to_trans = self.columns
            for clm in columns_to_trans:
                X[clm] = X[clm].apply(lambda x: str(x))
        return X.set_index('date')

app = FastAPI()
class Mydatatype(BaseModel):
    locale_name : object
    holiday : object
    transferred : object
    description : object
    locale : object
    dcoilwtico : float
    transactions : float
    onpromotion : int
    date : object
    family : object
    store_nbr : int
    city : object
    state : object
    store_type : object
    cluster : int
    id : int
model = pickle.load(open("pipeline.pkl","rb"))
@app.post("/")
async def forecasting_endpoint(data:Mydatatype):
    df = pd.DataFrame(data)
    return model.predict(data)
