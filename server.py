import signal
from flask import Flask, request
from flask_cors import CORS
import requests
import json
import pandas as pd
import numpy as np
from predict import predict
from requests.auth import HTTPBasicAuth
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math, time
from _thread import start_new_thread

scalers = {}
prices = {}
prediction_set = []
df = []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_data(stock, seq_len, split):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1
    result = []
    global scalers
    global prices
    
    for index in range(len(data) - sequence_length):
        scalers[index] = MinMaxScaler(feature_range=(0,1))
        prices[index] = MinMaxScaler(feature_range=(0,1))
        
        prices[index].fit_transform(data[index: index + sequence_length][:, -1].reshape(-1,1))
        result.append(scalers[index].fit_transform(data[index: index + sequence_length]))
    
    result = np.array(result)
    row = len(result) * split
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  
    
    return [x_train, y_train, x_test, y_test]

app = Flask(__name__)

Bearer = None

def prep_prediction(df, ticker, _min, _max):
    dates = df['date']
    labels = df['label']

    df = df.drop(['date', 'label'], axis=1)
    columnsTitles = ["change", "changeOverTime", "changePercent", "high", "low", "open", "unadjustedVolume", "volume", "vwap", "close"]
    df=df.reindex(columns=columnsTitles)

    Bearer = getBearer()
    headers = {
        'Authorization' : 'Bearer ' + Bearer,
        'Content-Type': 'application/json'
    }
    payload = {
        "startDate": _min,
        "endDate": _max,
        "where": {
            "ticker" : [ticker]
        }
    }
    response = requests.post('https://api.marquee.gs.com/v1/data/USCANFPP_MINI/query', json=payload, headers=headers)
    extraDf = pd.DataFrame(response.json()['data'])
    df['financialReturnsScore'] = extraDf['financialReturnsScore']
    df['growthScore'] = extraDf['growthScore']
    df['integratedScore'] = extraDf['integratedScore']
    df['multipleScore'] = extraDf['multipleScore']

    columnsTitles = ["change", "changeOverTime", "changePercent", "high", "low", "open", "unadjustedVolume", "volume", "vwap", "financialReturnsScore", "growthScore", "integratedScore", "multipleScore", "close"]
    df=df.reindex(columns=columnsTitles)

    window = 15 # Another hyperparameter

    X_train, y_train, X_test, y_test = load_data(df[::-1], window, 0.85)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    global prediction_set
    prediction_set = X_test

def getBearer():
    payload = "nothing to see here"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post('https://idfs.gs.com/as/token.oauth2', data=payload, headers=headers)
    print (response)
    return response.json()['access_token']

@app.route('/')
def index():
    return 'API working'

@app.route("/gs/all/")
def all():
    try:
        if Bearer is None:
            Bearer = getBearer()
    except UnboundLocalError:
        Bearer = getBearer()
    headers = {
        'Authorization' : 'Bearer ' + Bearer
    }
    response = requests.get('https://api.marquee.gs.com/v1/data/USCANFPP_MINI/coverage', headers=headers)
    return response.text

@app.route("/gs/<string:ticker>", methods=['POST'])
def gs(ticker):
    data = request.get_json()
    print (data, ticker)
    try:
        if Bearer is None:
            Bearer = getBearer()
    except UnboundLocalError:
        Bearer = getBearer()
    headers = {
        'Authorization' : 'Bearer ' + Bearer,
        'Content-Type': 'application/json'
    }
    payload = {
        "startDate": data['start'],
        "endDate": data['end'],
        "where": {
            "ticker" : [ticker]
        }
    }
    response = requests.post('https://api.marquee.gs.com/v1/data/USCANFPP_MINI/query', json=payload, headers=headers)
    
    return response.text

@app.route("/predict")
def pred():
    i = predict(prediction_set)
    results = []
    pred_results = []
    counter = 0
    for x in np.arange(len(df) - 137 - 15, len(df) - 15 - 1, 1):
        results = results + [prices[x].inverse_transform(i[counter: counter + 15]).reshape(-1,1)[0]]
        counter = counter + 1
    return json.dumps({"data":results}, cls=NumpyEncoder)

@app.route("/stock/<string:ticker>")
def stock(ticker):
    global df
    response = requests.get('https://api.iextrading.com/1.0/stock/'+ ticker +'/chart/5y')
    df = pd.DataFrame(response.json())

    df = df[df['date'] < '2017-06-28']
    min_date, max_date = df['date'].min(), df['date'].max()

    start_new_thread(prep_prediction, (df, ticker, min_date, max_date))
    return df.to_json(orient='records')

application = app #gunicorn looks for application
CORS(app)
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
