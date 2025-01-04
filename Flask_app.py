from flask import Flask, request, jsonify, render_template
import joblib  # To load the model
import numpy as np
from src.components.feature_engineering import feature_engineering
from sklearn.feat import RobustScaler  # To scale the features

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("artifacts/models/catboost_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            # Get data from form submission
            amount = float(request.form.get("amount"))
            oldbalanceOrg = float(request.form.get("Previous Balance in account for Sender"))
            oldbalanceDest = float(request.form.get("Previous Balance in account for Receiver"))
            step = float(request.form.get("step"))
            type_CASH_OUT = int(request.form.get("type_CASH_OUT"))
            type_TRANSFER = int(request.form.get("type_TRANSFER"))
            type_PAYMENT = int(request.form.get("type_PAYMENT"))
            type_DEBIT = int(request.form.get("type_DEBIT"))

            # Create feature array
            features = np.array([[amount, oldbalanceOrg, oldbalanceDest, step, type_CASH_OUT, type_TRANSFER, type_PAYMENT, type_DEBIT]])

            scaled_features= RobustScaler().fit_transform(features)
            # Make a prediction
            prediction = model.predict(scaled_features)
            result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        except Exception as e:
            result = f"Error: {str(e)}"

    # Render the form and display result on the same page
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
