
# from types import NoneType
from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import joblib
import json
from joblib import load
import pandas as pd
from input_v3 import Model
import yfinance as yf
# from keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import date


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
 # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        ticker = request.form.get("Ticker")
        SPF = Model()
        data = SPF.extract_data(ticker)
        # Get prediction
        stock_prediction = SPF.reshape_model()

    else:
        stock_prediction = ""

    return render_template("index.html", output = stock_prediction) 



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