
from flask import Flask, render_template, request
import numpy as np
import joblib
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# import uuid
from input_v2 import Model
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#load model
model = load_model("model.h5")

app = Flask(__name__)

# @app.route('/index')
# def index():
#     app.static_folder = 'static'
#     return render_template("index.html")

# @app.route('/getstarted')
# def getstarted():
#     app.static_folder = 'static'
#     return render_template("getstarted.html")

# @app.route('/crypto')
# def crypto():
#     app.static_folder = 'static'
#     return render_template("crypto.html")

# @app.route('/calendar')
# def calendar():
#     app.static_folder = 'static'
#     return render_template("calendar.html")

@app.route('/', methods=['GET', 'POST'])
def main():
    
 # If a form is submitted
    if request.method == "POST":
        
        
        # Get values through input bars
        ticker = request.form.get("Ticker")
        SPF = Model()
        data = SPF.extract_data(ticker)
        X = SPF.reshape()

        # scaler = MinMaxScaler(feature_range = (0, 1))
        # Get prediction
        prediction = model.predict(X)
        # prediction = scaler.inverse_transform(prediction)
        
    else:
        prediction = ""
        
    return render_template("index.html", output = prediction)

@app.route('/getstarted')
def getstarted():
    app.static_folder = 'static'
    return render_template("getstarted.html")

@app.route('/crypto')
def crypto():
    app.static_folder = 'static'
    return render_template("crypto.html")

@app.route('/calendar')
def calendar():
    app.static_folder = 'static'
    return render_template("calendar.html")

@app.route('/sp500')
def sp500():
    app.static_folder = 'static'
    return render_template("s&p500.html")


# Running the app
if __name__ == '__main__':
    app.run(debug = True)