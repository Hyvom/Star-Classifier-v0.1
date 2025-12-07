from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("star_model.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            float(data["temperature"]),
            float(data["luminosity"]),
            float(data["radius"]),
            float(data["absolute_magnitude"])
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Star type mapping (adjust based on your dataset labels)
        star_types = {
            0: "Brown Dwarf",
            1: "Red Dwarf",
            2: "White Dwarf",
            3: "Main Sequence",
            4: "Supergiant",
            5: "Hypergiant"
        }
        
        return jsonify({
            "star_type": star_types.get(prediction, "Unknown"),
            "star_type_code": int(prediction),
            "features": {
                "temperature": features[0],
                "luminosity": features[1],
                "radius": features[2],
                "absolute_magnitude": features[3]
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
