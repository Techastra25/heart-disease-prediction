"""
test_model.py
--------------
Pytest tests — data integrity, model correctness, API behaviour.
Run: pytest tests/ -v
"""

import pickle, json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score


@pytest.fixture(scope="module")
def model():
    with open("models/logistic_regression.pkl","rb") as f: return pickle.load(f)

@pytest.fixture(scope="module")
def scaler():
    with open("models/scaler.pkl","rb") as f: return pickle.load(f)

@pytest.fixture(scope="module")
def df():
    return pd.read_csv("data/heart.csv")

@pytest.fixture(scope="module")
def X_scaled(df, scaler):
    return scaler.transform(df.drop("target", axis=1))

@pytest.fixture(scope="module")
def flask_client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','src'))
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client: yield client


# ── Data Integrity ─────────────────────────────────────────
class TestData:
    def test_row_count(self, df):
        assert len(df) == 303

    def test_no_nulls(self, df):
        assert df.isnull().sum().sum() == 0

    def test_columns(self, df):
        expected = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal','target']
        assert list(df.columns) == expected

    def test_binary_target(self, df):
        assert set(df['target'].unique()) == {0, 1}


# ── Model Correctness ──────────────────────────────────────
class TestModel:
    def test_accuracy_above_90(self, model, df, X_scaled):
        acc = accuracy_score(df['target'], model.predict(X_scaled))
        assert acc > 0.90, f"Got {acc:.4f}"

    def test_roc_auc_above_95(self, model, df, X_scaled):
        auc = roc_auc_score(df['target'], model.predict_proba(X_scaled)[:,1])
        assert auc > 0.95, f"Got {auc:.4f}"

    def test_binary_output(self, model, df, X_scaled):
        assert set(model.predict(X_scaled)).issubset({0, 1})

    def test_probabilities_valid(self, model, df, X_scaled):
        probs = model.predict_proba(X_scaled)[:,1]
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_single_prediction_shape(self, model, scaler):
        patient = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
        pred    = model.predict(scaler.transform(patient))[0]
        assert pred in [0, 1]


# ── Flask API ──────────────────────────────────────────────
PATIENT = {"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,
           "fbs":0,"restecg":1,"thalach":168,"exang":0,
           "oldpeak":1.0,"slope":2,"ca":2,"thal":3}

class TestAPI:
    def test_health_ok(self, flask_client):
        r = flask_client.get("/health")
        assert r.status_code == 200
        assert json.loads(r.data)["status"] == "ok"

    def test_predict_200(self, flask_client):
        r = flask_client.post("/predict",
              data=json.dumps(PATIENT), content_type="application/json")
        assert r.status_code == 200

    def test_predict_keys(self, flask_client):
        r    = flask_client.post("/predict",
              data=json.dumps(PATIENT), content_type="application/json")
        body = json.loads(r.data)
        for k in ["prediction","diagnosis","probability","risk_level"]:
            assert k in body

    def test_predict_missing_features_400(self, flask_client):
        r = flask_client.post("/predict",
              data=json.dumps({"age":52}), content_type="application/json")
        assert r.status_code == 400

    def test_batch_count(self, flask_client):
        r    = flask_client.post("/predict/batch",
              data=json.dumps([PATIENT, PATIENT]), content_type="application/json")
        body = json.loads(r.data)
        assert body["total"] == 2 and len(body["results"]) == 2

    def test_risk_level_valid(self, flask_client):
        r    = flask_client.post("/predict",
              data=json.dumps(PATIENT), content_type="application/json")
        risk = json.loads(r.data)["risk_level"]
        assert risk in ["Low","Medium","High"]
