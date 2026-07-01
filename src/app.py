"""
app.py
-------
Flask REST API — Heart Disease Prediction.

Loads Logistic Regression model (95.08% accuracy, 98.92% ROC-AUC)
and StandardScaler at startup — not per request.

Endpoints:
  GET  /health          →  model status + metrics
  POST /predict         →  single patient prediction
  POST /predict/batch   →  batch predictions (list of patients)

Run:
    python src/app.py

Test:
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
           "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
           "slope":0,"ca":0,"thal":1}'
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load once at startup — not per request
with open("models/logistic_regression.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

FEATURES = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']

MODEL_META = {
    "name":     "Logistic Regression",
    "accuracy": "95.08%",
    "roc_auc":  "98.92%",
    "cv_score": "94.39% (5-fold StratifiedKFold)",
    "dataset":  "Cleveland Heart Disease — 303 patients, 13 features",
}


def _predict_one(record: dict) -> dict:
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    arr    = np.array([[record[f] for f in FEATURES]], dtype=float)
    scaled = SCALER.transform(arr)
    pred   = int(MODEL.predict(scaled)[0])
    prob   = float(MODEL.predict_proba(scaled)[0][1])
    return {
        "prediction":  pred,
        "diagnosis":   "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        "probability": round(prob, 4),
        "risk_level":  "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_META})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    try:
        return jsonify(_predict_one(data))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Body must be a non-empty JSON array"}), 400
    try:
        results = [_predict_one(r) for r in data]
        return jsonify({"results": results, "total": len(results)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Batch failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("\nHeart Disease Prediction API")
    print(f"Model   : Logistic Regression | Accuracy: 95.08% | AUC: 98.92%")
    print("Server  : http://localhost:5000")
    print("Endpoints: GET /health | POST /predict | POST /predict/batch\n")
    app.run(debug=False, port=5000)
