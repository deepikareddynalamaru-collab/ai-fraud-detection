"""
test_system.py
End-to-End System Integration and Robustness Test Suite.
"""

import pandas as pd
from src.predict import FraudPredictor

def run_tests():
    print("Running AI Fraud Detection Integration Test Suite...")
    pred = FraudPredictor()

    # Edge Case 1: Unknown User ID, unseen merchant category, high risk
    edge1 = {
        "user_id": "USR_99999_NEVER_SEEN",
        "amount": 500000.0,
        "merchant_category": "unseen_foreign_exchange",
        "city": "Tokyo",
        "device_type": "Linux Rig",
        "is_international": 1,
        "txn_count_last_1h": 8
    }
    r1 = pred.predict_one(edge1)
    assert r1["risk_level"] == "High", f"Expected High, got {r1['risk_level']}"
    assert len(r1["reasons"]) >= 1, "Expected at least 1 reason"
    print(f"✅ Test 1 (Unknown User + Extreme Anomaly): PASSED. Score={r1['risk_score']}, Reasons={len(r1['reasons'])}")

    # Edge Case 2: Minimal normal transaction
    edge2 = {
        "user_id": "USR_0005",
        "amount": 45.0,
        "merchant_category": "food_dining",
        "city": "Mumbai",
        "device_type": "Android Phone",
        "is_international": 0,
        "txn_count_last_1h": 1
    }
    r2 = pred.predict_one(edge2)
    assert r2["risk_level"] == "Low", f"Expected Low, got {r2['risk_level']}"
    assert len(r2["reasons"]) >= 1, "Expected at least 1 reason"
    print(f"✅ Test 2 (Routine Micro-Payment): PASSED. Score={r2['risk_score']}, Reasons={len(r2['reasons'])}")

    # Edge Case 3: Batch scoring with missing optional columns
    edge3_df = pd.DataFrame([
        {"amount": 150.0, "merchant_category": "grocery"},
        {"amount": 85000.0, "merchant_category": "crypto_investment", "city": "London", "is_international": 1}
    ])
    r3_df = pred.predict_batch(edge3_df)
    assert len(r3_df) == 2
    assert "risk_score" in r3_df.columns
    assert "risk_level" in r3_df.columns
    assert "primary_reason" in r3_df.columns
    print("✅ Test 3 (Batch scoring with missing optional schema): PASSED.")

    # Edge Case 4: Verify Success Metrics from saved training run
    import json
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    ens = metrics["ensemble"]
    assert ens["recall"] >= 0.85, f"Recall {ens['recall']} < 0.85"
    assert ens["precision"] >= 0.70, f"Precision {ens['precision']} < 0.70"
    assert ens["roc_auc"] >= 0.95, f"ROC-AUC {ens['roc_auc']} < 0.95"
    print(f"✅ Test 4 (Success Metrics Verified): Recall={ens['recall']:.4f}, Precision={ens['precision']:.4f}, ROC-AUC={ens['roc_auc']:.4f}")

    print("\nALL 4 INTEGRATION TESTS PASSED SUCCESSFULLY! 🎯")

if __name__ == "__main__":
    run_tests()

