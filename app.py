<<<<<<< HEAD
from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)

=======


from flask import Flask, render_template, request
import numpy as np
import joblib
# from joblib import load
import pandas as pd

# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
import uuid
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout

app = Flask(__name__)

@app.route('/index')
def index():
    app.static_folder = 'static'
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
>>>>>>> 64baa503ec738c0f2584164369c0fe07ec96f860

@app.route('/', methods=['GET', 'POST'])
def main():
    
<<<<<<< HEAD
=======
    
>>>>>>> 64baa503ec738c0f2584164369c0fe07ec96f860
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
<<<<<<< HEAD
        model = joblib.load("model.pkl")
=======
        model = joblib.load(model, "model.pkl")
>>>>>>> 64baa503ec738c0f2584164369c0fe07ec96f860
        
        # Get values through input bars
        ticker = request.form.get("Ticker")
       
        
        # Put inputs to dataframe
        X = pd.DataFrame([[ticker]], columns = ["Ticker"])
        
        # Get prediction
        prediction = model.predict(X)[0]
        
    else:
        prediction = ""
        
<<<<<<< HEAD
    return render_template("index.html", output = prediction)
=======
    return render_template("index.html")
>>>>>>> 64baa503ec738c0f2584164369c0fe07ec96f860


# Running the app
if __name__ == '__main__':
    app.run(debug = True)