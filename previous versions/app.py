

from flask import Flask, render_template, request
import numpy as np
import joblib
# from joblib import load
import pandas as pd
# import pickle
# from keras.models import Sequential
# from keras.models import load_model


# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
import uuid

#load model
# model = load_model("model.h5" )
# print("@@ model loaded")

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/index')
# def index():
#     app.static_folder = 'static'
#     return render_template("index.html")



@app.route('/', methods=['GET', 'POST'])
def main():
    
    app.static_folder = 'static'
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        # model = joblib.load("model.joblib")
        # model = pickle.load("model.pkl")
        

        
        # Get values through input bars
        ticker = request.form.get("Ticker")
       
        
        # Put inputs to dataframe
        X = pd.DataFrame([[ticker]], columns = ["Ticker"])
        
        # Get prediction
        prediction = model.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("index.html")

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
    app.run(debug = 1)