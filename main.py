from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class Plant(BaseModel):
    sepalLength : float
    sepalWidth: float
    petalLength: float
    petalWidth: float

def load_Model():
    model = joblib.load("xgboost_iris_model.joblib")
    return model

@app.get('/')
async def isWorking():
    return {"hello": "world"}

@app.post('/')
async def check(item:Plant):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    model = load_Model()
    yhat = model.predict(df)[0]
    return {"Prediction":int(yhat)}

