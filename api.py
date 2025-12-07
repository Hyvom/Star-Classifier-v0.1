from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json

# Load model
model = joblib.load("star_model.joblib")

# Load metrics
with open("metrics.json", "r") as f:
    METRICS = json.load(f)

app = Flask(__name__)

star_types = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/model")
def model_page():
    return render_template("model.html")

@app.route("/stars")
def stars_page():
    return render_template("stars.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data["temperature"]),
            float(data["luminosity"]),
            float(data["radius"]),
            float(data["absolute_magnitude"])
        ]

        # Prediction
        probs = model.predict_proba([features])[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Per-class metrics from metrics.json
        class_report = METRICS["report"].get(str(pred_class), {})
        precision = class_report.get("precision", None)
        recall = class_report.get("recall", None)
        f1 = class_report.get("f1-score", None)

        return jsonify({
            "star_type": star_types.get(pred_class, "Unknown"),
            "star_type_code": pred_class,
            "confidence": confidence,
            "metrics": {
                "overall_accuracy": METRICS["accuracy"],
                "class_precision": precision,
                "class_recall": recall,
                "class_f1": f1
            },
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
