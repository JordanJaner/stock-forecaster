from flask import Flask, render_template, request
import pickle
import numpy as np
from joblib import load

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == "POST":
        model = pickle.load(open('lr_model.pkl', 'rb'))
        user_input = request.form.get('size')
        user_input = float(user_input)
        prediction = model.predict([[user_input]])
        print(prediction)
    return render_template('index.html', prediction=prediction)

    # test_np_input = np.array([[1]], [[2]], [[17]])
    # model = load('model.joblib')
    # preds = model.predict(test_np_input)
    # preds_as_str = str(preds)
    # return preds_as_str

if __name__ == '__main__':
    app.run(debug=True)
