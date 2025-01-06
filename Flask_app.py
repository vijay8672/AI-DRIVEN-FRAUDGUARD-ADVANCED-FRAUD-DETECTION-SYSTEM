from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("artifacts/models/adaboost_model.pkl")
scaler = joblib.load("artifacts/model_artifacts/scaler.pkl")  # Load the saved scaler

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            # Get data from the form submission
            type_CASH_OUT = float(request.form.get("type_CASH_OUT", 0))
            type_PAYMENT = float(request.form.get("type_PAYMENT", 0))
            type_TRANSFER = float(request.form.get("type_TRANSFER", 0))
            amount = float(request.form.get("amount"))
            oldbalanceOrg = float(request.form.get("oldbalanceOrg"))
            oldbalanceDest = float(request.form.get("oldbalanceDest"))

            # Create feature array
            features = np.array([[type_CASH_OUT, type_PAYMENT, type_TRANSFER, amount, oldbalanceOrg, oldbalanceDest]])

            # Scale the features using the loaded scaler
            scaled_features = scaler.transform(features)

            # Make a prediction
            prediction = model.predict(scaled_features)
            result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        except Exception as e:
            result = f"Error: {str(e)}"

    # Render the form and display the result on the same page
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
