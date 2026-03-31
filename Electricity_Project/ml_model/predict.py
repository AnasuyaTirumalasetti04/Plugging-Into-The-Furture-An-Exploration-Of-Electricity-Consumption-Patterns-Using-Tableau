from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initialize Flask
app = Flask(__name__, template_folder="../templates")
CORS(app)

# Load dataset
data = pd.read_csv("../data/Consumption.csv")

# Convert Dates column 
data["Dates"] = pd.to_datetime(data["Dates"], dayfirst=True)

# Convert date into numeric value for ML
data["day"] = data["Dates"].map(pd.Timestamp.toordinal)

# Train a model for each state
models = {}

for state in data["States"].unique():

    df = data[data["States"] == state]

    X = df[["day"]]
    y = df["Usage"]

    model = LinearRegression()
    model.fit(X, y)

    models[state] = model


# Home page
@app.route("/")
def home():
    return render_template("prediction.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:

        req = request.get_json()

        if not req:
            return jsonify({"error": "Invalid request"}), 400

        state = req.get("state")
        date = req.get("date")

        if not state or not date:
            return jsonify({"error": "Missing input"}), 400

        # Convert date
        date = pd.to_datetime(date, dayfirst=True)
        day = date.toordinal()

        # Normalize state name
        state_clean = state.strip().lower()

        # Check if real data exists
        result = data[
            (data["States"].str.lower() == state_clean) &
            (data["Dates"] == date)
        ]

        if not result.empty:
            usage = float(result["Usage"].values[0])

        else:

            # Use ML model
            model = models.get(state.title())

            if model is None:
                return jsonify({"error": "State not found"}), 404

            usage = float(model.predict([[day]])[0])

        return jsonify({
            "usage": usage
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": "Server error"
        }), 500


# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)