"""
train.py
---------
End-to-end ML training pipeline — Cleveland Heart Disease Dataset.

Steps:
  1. Load & validate data (null check, type check, class distribution)
  2. Stratified 80/20 split — preserves class ratio on small dataset
  3. StandardScaler fit on train ONLY — prevents data leakage
  4. Train Logistic Regression + Random Forest
  5. Evaluate: Accuracy, ROC-AUC, 5-fold StratifiedKFold CV
  6. Save best model + scaler as pickle
  7. Generate 4 result charts

Run from project root:
    python src/train.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

DATA_PATH  = "data/heart.csv"
MODEL_DIR  = "models"
IMAGES_DIR = "docs/images"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_and_validate(path):
    df = pd.read_csv(path)
    print(f"[data] rows={len(df)} | features={df.shape[1]-1} | nulls={df.isnull().sum().sum()}")
    print(f"[data] class distribution:\n{df['target'].value_counts()}\n")
    assert df.isnull().sum().sum() == 0,         "Dataset has nulls — fix before training"
    assert set(df['target'].unique()) == {0, 1}, "Target must be binary 0/1"
    return df


def preprocess(df):
    X = df.drop('target', axis=1)
    y = df['target']
    # stratify=y → preserves 54/46 disease ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)  # fit ONLY on train
    X_test_sc  = scaler.transform(X_test)       # apply train stats to test
    print(f"[split] train={len(X_train)} | test={len(X_test)}")
    return X, y, X_train_sc, X_test_sc, y_train, y_test, scaler


def train_evaluate(X, y, X_train_sc, X_test_sc, y_train, y_test, scaler):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)
        prob = model.predict_proba(X_test_sc)[:, 1]
        acc  = accuracy_score(y_test, pred)
        auc  = roc_auc_score(y_test, prob)
        # StratifiedKFold → class ratio preserved in every fold
        cv   = cross_val_score(
            model, scaler.transform(X), y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        ).mean()
        results[name] = {
            "model": model, "pred": pred, "prob": prob,
            "accuracy": acc, "roc_auc": auc, "cv_score": cv
        }
        print(f"\n{'='*50}\n[{name}]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  CV Mean  : {cv:.4f}")
        print(f"\n{classification_report(y_test, pred, target_names=['No Disease','Disease'])}")
    return results


def save_models(results, scaler):
    for name, r in results.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        with open(f"{MODEL_DIR}/{fname}", "wb") as f:
            pickle.dump(r["model"], f)
        print(f"[save] {fname}")
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[save] scaler.pkl")


def generate_charts(df, results, y_test):
    items = list(results.items())

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, r) in zip(axes, items):
        cm = confusion_matrix(y_test, r["pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease','Disease'],
                    yticklabels=['No Disease','Disease'])
        ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.2%}", fontsize=12)
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    fig.suptitle("Confusion Matrices — Heart Disease Classification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/confusion_matrices.png", dpi=130, bbox_inches='tight')
    plt.close()

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, r), color in zip(items, ['#2563eb', '#16a34a']):
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{name}  (AUC = {r['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],'--', color='#9ca3af', lw=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Heart Disease Prediction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/roc_curves.png", dpi=130, bbox_inches='tight')
    plt.close()

    # 3. Feature Importance (Random Forest)
    rf = results["Random Forest"]["model"]
    fi = pd.DataFrame({'feature': df.drop('target',axis=1).columns,
                       'importance': rf.feature_importances_}
                     ).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#93c5fd' if i < len(fi)-3 else '#2563eb' for i in range(len(fi))]
    ax.barh(fi['feature'], fi['importance'], color=colors)
    ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/feature_importance.png", dpi=130, bbox_inches='tight')
    plt.close()

    # 4. Age & Heart Rate Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for val, label, color in [(0,'No Disease','#16a34a'), (1,'Heart Disease','#dc2626')]:
        sub = df[df['target']==val]
        axes[0].hist(sub['age'],     alpha=0.65, label=label, color=color, bins=15, edgecolor='white')
        axes[1].hist(sub['thalach'], alpha=0.65, label=label, color=color, bins=15, edgecolor='white')
    for ax, title, xlabel in zip(axes,
            ['Age Distribution by Diagnosis', 'Max Heart Rate by Diagnosis'],
            ['Age', 'Max Heart Rate (thalach)']):
        ax.set_title(title, fontsize=12); ax.set_xlabel(xlabel)
        ax.set_ylabel('Count'); ax.legend()
    fig.suptitle("Clinical Feature Distributions — Cleveland Dataset", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/distributions.png", dpi=130, bbox_inches='tight')
    plt.close()

    print(f"\n[charts] 4 charts → {IMAGES_DIR}/")


def main():
    df = load_and_validate(DATA_PATH)
    X, y, X_train_sc, X_test_sc, y_train, y_test, scaler = preprocess(df)
    results = train_evaluate(X, y, X_train_sc, X_test_sc, y_train, y_test, scaler)
    save_models(results, scaler)
    generate_charts(df, results, y_test)

    os.makedirs("docs", exist_ok=True)
    summary = pd.DataFrame({
        'Model':    list(results.keys()),
        'Accuracy': [round(r['accuracy'],4) for r in results.values()],
        'ROC_AUC':  [round(r['roc_auc'], 4) for r in results.values()],
        'CV_Score': [round(r['cv_score'],4) for r in results.values()],
    })
    summary.to_csv('docs/model_results.csv', index=False)

    print("\n" + "="*55)
    print("FINAL MODEL COMPARISON")
    print("="*55)
    print(summary.to_string(index=False))
    best = summary.loc[summary['Accuracy'].idxmax(), 'Model']
    print(f"\n→ Best model: {best} → deployed in Flask API")


if __name__ == "__main__":
    main()
