import pickle
from flask import Flask, request, json, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application
print(application)
print(__name__)
## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open("models/ridge.pkl", 'rb'))
scaler = pickle.load(open("models/scaler.pkl", 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        print(request.form)
        Temperature = float(request.form.get("Temperature" ))
        Ws = float(request.form.get("Ws"))
        RH = float(request.form.get("RH"))
        Rainfall = float(request.form.get("Rainfall"))
        DWC = float(request.form.get("DWC"))
        FFMC = float(request.form.get("FFMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_data = scaler.transform([[Temperature, RH, Ws, Rainfall, DWC, FFMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data)

        return render_template("home.html", results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
