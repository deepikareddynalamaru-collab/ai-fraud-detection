# 🛡️ AI-Powered Fraud Detection System
>  AI for Finance  
> **Tech Stack:** Python 3.13, scikit-learn, Streamlit, Plotly, Pandas, NumPy

An end-to-end, production-ready AI Fraud Detection System that identifies fraudulent financial transactions using a **Hybrid AI Architecture** (Unsupervised Isolation Forest + Supervised Balanced Random Forest), assigns a dynamic risk score (0–100), and provides **plain-English diagnostic explanations** for every flagged transaction.

---

## 🚀 Key Features

- **Hybrid AI Scoring Engine:** Fuses unsupervised zero-day anomaly scores (40%) and supervised fraud probabilities (60%) into a unified 0–100 risk score.
- **Three-Tier Risk Classification:**
  - 🟢 **Low Risk (0–39):** Auto-approved routine transactions.
  - 🟡 **Medium Risk (40–69):** Step-up authentication / OTP verification recommended.
  - 🔴 **High Risk (70–100):** Immediate transaction halt / priority analyst investigation.
- **Explainable AI (XAI):** Plain-English reasoning for every flag (e.g., *"Amount is 14.0x higher than user's average"*, *"Transaction made at 3 AM"*, *"Unrecognized device"*).
- **Interactive 4-Tab Streamlit Dashboard:**
  1. **Executive Dashboard:** Portfolio KPIs, amount at risk, fraud by hour, and category breakdown.
  2. **Check a Transaction:** Real-time form with risk gauge, sub-model breakdown, and 1-click fraud scenario presets.
  3. **Bulk Analysis:** Batch scoring with CSV upload, risk filtering, deep-dive transaction inspection, and CSV export.
  4. **Model Performance:** Confusion matrix, ROC curve, feature importances, and PRD metric verification.
- **Ultra-Fast & Lightweight:** Runs entirely on standard CPU hardware with no GPU required. Trains in ~10 seconds and scores 1,000 transactions in ~1.2 seconds.

---

## 📊 Benchmark Results (Tested on 4,018 Unseen Records)

| Metric | PRD Target | Achieved (Hybrid AI) | Status |
|---|---|---|---|
| **Recall (Fraud Class)** | $\ge 85.0\%$ | **98.98%** (97 of 98 caught) | ✅ Exceeded |
| **Precision (Fraud Class)** | $\ge 70.0\%$ | **97.98%** (97 of 99 true) | ✅ Exceeded |
| **ROC-AUC** | $\ge 0.950$ | **1.0000** | ✅ Perfect |
| **F1-Score** | Metric balance | **0.9848** | ✅ High balance |
| **False Positive Rate** | Minimal | **0.05%** (2 out of 3,920 normal) | ✅ Frictionless |
| **Scoring Latency** | $< 2.0\text{s / 1,000}$ | **1.277 seconds** | ✅ Real-time |

---

## 📁 Project Structure

```text
AI-Detection/
├── data/
│   ├── generate_data.py       # Synthetic data generator (20,000 txns, 6 fraud patterns)
│   └── transactions.csv       # Generated transaction dataset
├── src/
│   ├── __init__.py            # Python package initialization
│   ├── features.py            # Feature engineering (behavioral z-scores, velocity, nocturnal)
│   ├── train.py               # Model training (Isolation Forest + Balanced Random Forest)
│   └── predict.py             # Risk scoring & plain-English explanation engine
├── models/
│   ├── isolation_forest.joblib# Serialized unsupervised model
│   ├── random_forest.joblib   # Serialized supervised model
│   ├── feature_engineer.joblib# Learned user behavioral profiles & encoders
│   ├── anomaly_scaler.joblib  # Anomaly score normalizer
│   ├── metrics.json           # Benchmark evaluation results
│   └── test_predictions.csv   # Pre-scored 4,018 test transactions
├── report/
│   └── REPORT.md              # Detailed academic & technical evaluation report
├── app.py                     # 4-Tab Streamlit web application
├── requirements.txt           # Python library dependencies
├── test_system.py             # Integration test suite
└── README.md                  # Setup guide and documentation
```

---

## ⚡ Quick Start: 3 Commands to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-generate Data & Train Models
*The repository already includes pre-trained models and sample data.* If you wish to re-train from scratch:
```bash
python data/generate_data.py
python src/train.py
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Running Integration Tests
To verify all models, edge cases (unknown users, missing fields, micro-payments), and PRD success criteria:
```bash
python test_system.py
```

---

## 🎙️ Short Demo Presentation Script (for Evaluators)

| Step | Action | What to Say / Highlight |
|---|---|---|
| **1. Overview** | Open **Dashboard tab** | *"Digital payment fraud is growing faster than humans can review. We built a dual-AI detection system combining Unsupervised Isolation Forest and Supervised Balanced Random Forest."* |
| **2. Portfolio KPIs** | Point to top metric cards | *"On our 20,000 transaction dataset, the system tracks 490 fraud cases representing ₹26.5M at risk, revealing a clear fraud spike during nocturnal hours (1–4 AM)."* |
| **3. Live Check** | Go to **Check a Transaction tab** | *"Let's test an incoming payment. I'll click the preset '🔴 3 AM Nocturnal Crypto'."* |
| **4. AI Explainability** | Review the Gauge & Explanations | *"Notice the risk score jumps to 100/100 (High Risk). Crucially, the AI is NOT a black box: it explains in plain English that the amount is 14x the user's average, occurred at 3 AM, used a new device, and had an abrupt city shift."* |
| **5. Bulk Batching** | Open **Bulk Analysis tab** | *"Fraud analysts can upload entire CSV batches or click 'Load Benchmark Test Batch' to review 4,018 transactions in ~1.2 seconds, sorted by risk level with one-click CSV export."* |
| **6. Performance** | Open **Model Performance tab** | *"On unseen test data, our hybrid model achieved 98.98% Recall (beating the 85% target), 97.98% Precision (beating the 70% target), and an ROC-AUC of 1.0000, ensuring near-zero false alarms."* |

---

## 📜 Academic Integrity & Course Context
Developed for **AI for Finance (CIA-2)**. Demonstrates practical machine learning for risk mitigation, automated feature engineering, and human-in-the-loop explainability for banking and fintech operations.

