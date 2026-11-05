"""
src/train.py
Training pipeline for AI-Powered Fraud Detection System.
Trains:
1. Isolation Forest (Unsupervised)
2. Random Forest (Supervised with class_weight='balanced')
Evaluates on 20% stratified test set and exports models and metrics.
"""

import os
import sys
import json
import joblib

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

from src.features import FeatureEngineer


def train_and_evaluate(data_path="data/transactions.csv", output_dir="models"):
    """
    Train Isolation Forest and Random Forest models and save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("AI FRAUD DETECTION SYSTEM: TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print(f"\n1. Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Total records: {len(df):,}")
    print(f"   Fraud records: {df['is_fraud'].sum():,} ({df['is_fraud'].mean() * 100:.2f}%)")
    
    # 2. Stratified Train/Test Split (80/20)
    print("\n2. Splitting into Train (80%) and Test (20%) using Stratified Sampling...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["is_fraud"]
    )
    # Reset indices to ensure clean alignment
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"   Train samples: {len(train_df):,} (Frauds: {train_df['is_fraud'].sum():,})")
    print(f"   Test samples:  {len(test_df):,} (Frauds: {test_df['is_fraud'].sum():,})")
    
    # 3. Fit Feature Engineer on Train set
    print("\n3. Fitting Feature Engineer on training transactions...")
    fe = FeatureEngineer()
    fe.fit(train_df)
    
    X_train = fe.transform(train_df)
    y_train = train_df["is_fraud"].values
    
    X_test = fe.transform(test_df)
    y_test = test_df["is_fraud"].values
    
    print(f"   Engineered {X_train.shape[1]} features:")
    for i, col in enumerate(X_train.columns, 1):
        print(f"     {i:2d}. {col}")
        
    # 4. Train Unsupervised Model: Isolation Forest
    print("\n4. Training Unsupervised Model: Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.025,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # Raw decision function: negative means anomaly, positive means normal
    # Invert so higher value means higher anomaly
    raw_train_anomaly = -iso_forest.decision_function(X_train)
    anomaly_scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    anomaly_scaler.fit(raw_train_anomaly.reshape(-1, 1))
    
    # 5. Train Supervised Model: Random Forest
    print("\n5. Training Supervised Model: Random Forest (Balanced)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # 6. Evaluate Models on Test Set
    print("\n6. Evaluating on Unseen Test Set (20%)...")
    
    # Unsupervised anomaly scores
    raw_test_anomaly = -iso_forest.decision_function(X_test)
    test_anomaly_scores = anomaly_scaler.transform(raw_test_anomaly.reshape(-1, 1)).flatten()
    
    # Supervised fraud probabilities
    test_rf_probs = rf.predict_proba(X_test)[:, 1]
    
    # Blended Risk Score = 0.4 * anomaly_score + 0.6 * fraud_probability, scaled 0-100
    test_risk_scores = (0.4 * test_anomaly_scores + 0.6 * test_rf_probs) * 100.0
    test_risk_scores = np.clip(test_risk_scores, 0.0, 100.0)
    
    # Binary predictions (threshold at 45.0 for balanced risk detection)
    test_ensemble_preds = (test_risk_scores >= 45.0).astype(int)
    test_rf_preds = rf.predict(X_test)
    
    # Ensemble metrics
    ens_precision = precision_score(y_test, test_ensemble_preds)
    ens_recall = recall_score(y_test, test_ensemble_preds)
    ens_f1 = f1_score(y_test, test_ensemble_preds)
    ens_roc_auc = roc_auc_score(y_test, test_risk_scores / 100.0)
    ens_cm = confusion_matrix(y_test, test_ensemble_preds).tolist()
    
    # Random Forest metrics
    rf_precision = precision_score(y_test, test_rf_preds)
    rf_recall = recall_score(y_test, test_rf_preds)
    rf_f1 = f1_score(y_test, test_rf_preds)
    rf_roc_auc = roc_auc_score(y_test, test_rf_probs)
    rf_cm = confusion_matrix(y_test, test_rf_preds).tolist()
    
    # Isolation Forest alone metrics
    iso_binary = (test_anomaly_scores >= 0.60).astype(int)
    iso_precision = precision_score(y_test, iso_binary, zero_division=0)
    iso_recall = recall_score(y_test, iso_binary, zero_division=0)
    iso_f1 = f1_score(y_test, iso_binary, zero_division=0)
    iso_roc_auc = roc_auc_score(y_test, test_anomaly_scores)
    
    # ROC Curve coordinates
    fpr, tpr, thresholds = roc_curve(y_test, test_risk_scores / 100.0)
    stride = max(1, len(fpr) // 100)
    roc_points = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in range(0, len(fpr), stride)]
    if roc_points[-1]["fpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})
        
    # Feature importances from Random Forest
    feature_importances = sorted(
        [{"feature": col, "importance": float(imp)} for col, imp in zip(X_train.columns, rf.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 50)
    print("Ensemble (Hybrid AI):")
    print(f"  - Precision: {ens_precision:.4f} (PRD Target >= 0.70)")
    print(f"  - Recall:    {ens_recall:.4f} (PRD Target >= 0.85)")
    print(f"  - F1-Score:  {ens_f1:.4f}")
    print(f"  - ROC-AUC:   {ens_roc_auc:.4f} (PRD Target >= 0.95)")
    print(f"  - Confusion Matrix:\n    TN: {ens_cm[0][0]}, FP: {ens_cm[0][1]}\n    FN: {ens_cm[1][0]}, TP: {ens_cm[1][1]}")
    
    print("\nRandom Forest (Supervised alone):")
    print(f"  - Precision: {rf_precision:.4f} | Recall: {rf_recall:.4f} | ROC-AUC: {rf_roc_auc:.4f}")
    
    print("\nIsolation Forest (Unsupervised alone):")
    print(f"  - Precision: {iso_precision:.4f} | Recall: {iso_recall:.4f} | ROC-AUC: {iso_roc_auc:.4f}")
    
    # Risk Level distribution on test set
    risk_levels = pd.cut(
        test_risk_scores,
        bins=[-1, 39.999, 69.999, 100.0],
        labels=["Low", "Medium", "High"]
    )
    risk_level_counts = risk_levels.value_counts().to_dict()
    print(f"\nRisk Level Distribution in Test Data: {risk_level_counts}")
    
    # 7. Save Models and Metadata
    print("\n7. Saving trained models and artifacts to directory...")
    joblib.dump(fe, os.path.join(output_dir, "feature_engineer.joblib"))
    joblib.dump(iso_forest, os.path.join(output_dir, "isolation_forest.joblib"))
    joblib.dump(rf, os.path.join(output_dir, "random_forest.joblib"))
    joblib.dump(anomaly_scaler, os.path.join(output_dir, "anomaly_scaler.joblib"))
    
    metrics = {
        "dataset": {
            "total_transactions": int(len(df)),
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "fraud_count_total": int(df["is_fraud"].sum()),
            "fraud_ratio_total": float(df["is_fraud"].mean()),
            "fraud_count_test": int(y_test.sum())
        },
        "ensemble": {
            "precision": round(float(ens_precision), 4),
            "recall": round(float(ens_recall), 4),
            "f1": round(float(ens_f1), 4),
            "roc_auc": round(float(ens_roc_auc), 4),
            "confusion_matrix": ens_cm
        },
        "random_forest": {
            "precision": round(float(rf_precision), 4),
            "recall": round(float(rf_recall), 4),
            "f1": round(float(rf_f1), 4),
            "roc_auc": round(float(rf_roc_auc), 4),
            "confusion_matrix": rf_cm
        },
        "isolation_forest": {
            "precision": round(float(iso_precision), 4),
            "recall": round(float(iso_recall), 4),
            "f1": round(float(iso_f1), 4),
            "roc_auc": round(float(iso_roc_auc), 4)
        },
        "roc_curve": roc_points,
        "feature_importance": feature_importances,
        "risk_level_counts": {str(k): int(v) for k, v in risk_level_counts.items()}
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to {metrics_path}")
    
    # Save test predictions for fast loading
    test_eval_df = test_df.copy().reset_index(drop=True)
    test_eval_df["anomaly_score"] = np.round(test_anomaly_scores, 4)
    test_eval_df["rf_fraud_prob"] = np.round(test_rf_probs, 4)
    test_eval_df["risk_score"] = np.round(test_risk_scores, 2)
    test_eval_df["risk_level"] = list(risk_levels)
    test_eval_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"   Saved test predictions to {os.path.join(output_dir, 'test_predictions.csv')}")
    
    print("\nTraining & evaluation completed successfully!")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
