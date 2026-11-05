"""
src/predict.py
Inference & Plain-English Explanation Engine for AI Fraud Detection System.
Provides single-transaction and batch scoring with risk levels and explanations.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import FeatureEngineer


class FraudPredictor:
    """
    Loads trained Isolation Forest, Random Forest, and Feature Engineer
    to predict risk scores, risk levels, and plain-English explanations.
    """

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.fe: FeatureEngineer = None
        self.iso_forest = None
        self.rf = None
        self.anomaly_scaler = None
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        """
        Load serialized model artifacts from disk.
        """
        fe_path = os.path.join(self.models_dir, "feature_engineer.joblib")
        iso_path = os.path.join(self.models_dir, "isolation_forest.joblib")
        rf_path = os.path.join(self.models_dir, "random_forest.joblib")
        scaler_path = os.path.join(self.models_dir, "anomaly_scaler.joblib")

        if not (os.path.exists(fe_path) and os.path.exists(iso_path) and os.path.exists(rf_path)):
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.models_dir}'. Run 'python src/train.py' first."
            )

        self.fe = joblib.load(fe_path)
        self.iso_forest = joblib.load(iso_path)
        self.rf = joblib.load(rf_path)
        self.anomaly_scaler = joblib.load(scaler_path)
        self.is_loaded = True

    def _determine_risk_level(self, risk_score: float) -> str:
        """
        FR3.4: Risk levels: Low (0-39), Medium (40-69), High (70-100).
        """
        if risk_score >= 70.0:
            return "High"
        elif risk_score >= 40.0:
            return "Medium"
        else:
            return "Low"

    def explain_transaction(self, feat_row: pd.Series, raw_dict: dict, risk_score: float) -> list:
        """
        FR4: Plain-English Explainability Engine.
        Returns 1-4 plain-English reasons for flagged transactions.
        Guarantees that every flagged transaction has at least 1 reason.
        """
        reasons = []

        def safe_float(v, default=0.0):
            try:
                return default if pd.isna(v) else float(v)
            except Exception:
                return default

        def safe_int(v, default=0):
            try:
                return default if pd.isna(v) else int(v)
            except Exception:
                return default

        amount = safe_float(raw_dict.get("amount", feat_row.get("amount", 0.0)))
        user_id = str(raw_dict.get("user_id", ""))
        user_avg = None
        if self.fe and user_id in self.fe.user_stats:
            user_avg = self.fe.user_stats[user_id]["mean_amount"]

        ratio = safe_float(feat_row.get("amount_ratio", 1.0), 1.0)
        zscore = safe_float(feat_row.get("amount_zscore", 0.0), 0.0)
        hour = safe_int(feat_row.get("hour_of_day", 12), 12)
        is_night = safe_int(feat_row.get("is_night", 0), 0)
        txn_count = safe_int(feat_row.get("txn_count_last_1h", 1), 1)
        loc_changed = safe_int(feat_row.get("location_changed", 0), 0)
        new_dev = safe_int(feat_row.get("is_new_device", 0), 0)
        is_intl = safe_int(raw_dict.get("is_international", feat_row.get("is_international", 0)), 0)
        category = str(raw_dict.get("merchant_category", ""))
        city = str(raw_dict.get("city", "Unknown"))
        device = str(raw_dict.get("device_type", "Unknown"))

        # Flagged transaction reasons (Medium or High Risk)
        if risk_score >= 40.0:
            # 1. Unusually high amount vs user baseline
            if ratio >= 2.5 or zscore >= 2.5:
                if user_avg and user_avg > 0:
                    reasons.append(
                        f"Amount is {ratio:.1f}x higher than this user's average (Rs. {amount:,.2f} vs avg Rs. {user_avg:,.2f})"
                    )
                else:
                    reasons.append(f"Amount of Rs. {amount:,.2f} is significantly higher than historical benchmarks")
            elif amount >= 25000.0:
                reasons.append(f"Unusually large single transaction volume (Rs. {amount:,.2f})")

            # 2. Odd hours / Nocturnal execution (1 - 4 AM)
            if is_night == 1 or (1 <= hour <= 4):
                reasons.append(f"Transaction made at {hour:02d}:00 hours (unusual nocturnal activity between 1 AM - 4 AM)")

            # 3. New or unrecognized device
            if new_dev == 1:
                reasons.append(f"New or unrecognized device used ('{device}')")

            # 4. Sudden location change / velocity anomaly
            if loc_changed == 1:
                reasons.append(f"Sudden city change detected ('{city}' differs from user's recent location)")

            # 5. Rapid burst frequency in last hour
            if txn_count >= 3:
                reasons.append(f"High velocity: {txn_count} transactions initiated within a 1-hour window")

            # 6. International transaction
            if is_intl == 1:
                if amount > 15000:
                    reasons.append(f"High-value cross-border international transaction in {city}")
                else:
                    reasons.append(f"Cross-border international transaction flagged for scrutiny ({city})")

            # 7. High-risk merchant category
            if category in ["crypto_investment", "money_transfer"]:
                reasons.append(f"High-risk merchant category: '{category.replace('_', ' ').title()}'")

            # Fallback if composite score tripped without individual extreme
            if not reasons:
                reasons.append(f"Composite behavioral anomaly flagged by hybrid AI models (Risk Score: {risk_score:.1f})")

        else:
            # Low Risk Transactions (Conforming to normal patterns)
            reasons.append("Standard transaction conforming to user's typical spending patterns and device history")
            if is_intl == 1:
                reasons.append("Low-value international purchase within acceptable risk parameters")

        return reasons[:4]

    def predict_one(self, txn_dict: dict) -> dict:
        """
        Score a single transaction from user input or manual check.
        """
        input_df = pd.DataFrame([txn_dict])
        
        # Ensure timestamp exists
        if "timestamp" not in input_df.columns or pd.isna(input_df["timestamp"].iloc[0]):
            input_df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Transform features
        X = self.fe.transform(input_df)

        # Unsupervised Isolation Forest Anomaly Score
        raw_anomaly = -self.iso_forest.decision_function(X)
        anomaly_score = float(self.anomaly_scaler.transform(raw_anomaly.reshape(-1, 1))[0][0])
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        # Supervised Random Forest Fraud Probability
        rf_prob = float(self.rf.predict_proba(X)[0][1])

        # FR3.3 Blended Risk Score = 0.4 * anomaly + 0.6 * rf_prob, scaled 0-100
        risk_score = round(float(np.clip((0.4 * anomaly_score + 0.6 * rf_prob) * 100.0, 0.0, 100.0)), 1)
        risk_level = self._determine_risk_level(risk_score)

        # Explainability
        feat_row = X.iloc[0]
        reasons = self.explain_transaction(feat_row, txn_dict, risk_score)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "anomaly_score": round(anomaly_score, 4),
            "fraud_probability": round(rf_prob, 4),
            "reasons": reasons,
            "primary_reason": reasons[0] if reasons else "Normal transaction",
            "engineered_features": {
                "amount": float(feat_row["amount"]),
                "amount_zscore": round(float(feat_row["amount_zscore"]), 2),
                "amount_ratio": round(float(feat_row["amount_ratio"]), 2),
                "hour_of_day": int(feat_row["hour_of_day"]),
                "is_night": int(feat_row["is_night"]),
                "txn_count_last_1h": int(feat_row["txn_count_last_1h"]),
                "location_changed": int(feat_row["location_changed"]),
                "is_new_device": int(feat_row["is_new_device"]),
                "is_international": int(feat_row["is_international"]),
            }
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch DataFrame of transactions.
        Appends risk_score, risk_level, anomaly_score, rf_probability, primary_reason, and reasons.
        """
        data = df.copy()
        
        # Transform features
        X = self.fe.transform(data)

        # Scores
        raw_anomaly = -self.iso_forest.decision_function(X)
        anomaly_scores = self.anomaly_scaler.transform(raw_anomaly.reshape(-1, 1)).flatten()
        anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)

        rf_probs = self.rf.predict_proba(X)[:, 1]

        risk_scores = np.clip((0.4 * anomaly_scores + 0.6 * rf_probs) * 100.0, 0.0, 100.0)
        risk_scores = np.round(risk_scores, 1)

        risk_levels = [self._determine_risk_level(s) for s in risk_scores]

        # Generate reasons for each transaction
        reasons_list = []
        primary_reasons = []

        for i in range(len(data)):
            raw_dict = data.iloc[i].to_dict()
            feat_row = X.iloc[i]
            r = self.explain_transaction(feat_row, raw_dict, risk_scores[i])
            reasons_list.append(r)
            primary_reasons.append(r[0] if r else "Normal transaction")

        result_df = data.copy()
        result_df["anomaly_score"] = np.round(anomaly_scores, 3)
        result_df["fraud_probability"] = np.round(rf_probs, 3)
        result_df["risk_score"] = risk_scores
        result_df["risk_level"] = risk_levels
        result_df["primary_reason"] = primary_reasons
        result_df["reasons"] = reasons_list

        return result_df
