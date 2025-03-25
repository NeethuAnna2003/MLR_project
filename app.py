from flask import Flask, render_template, request
import joblib  # Use joblib instead of pickle
import numpy as np

# Load trained model using joblib
model = joblib.load("house_price_model.joblib")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        try:
            # Get form data
            square_feet = float(request.form["square_feet"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            distance = float(request.form["distance"])

            # Predict price
            features = np.array([[square_feet, bedrooms, bathrooms, distance]])
            predicted_price = model.predict(features)[0]

        except Exception as e:
            predicted_price = f"Error: {e}"

    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
