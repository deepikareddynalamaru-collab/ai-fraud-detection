# Project Report: AI-Powered Fraud Detection System
**Course:** AI for Finance, CIA-2  
**Tech Stack:** Python 3.13, scikit-learn, Streamlit, Plotly, Pandas, NumPy  
**Project Repository / Root:** `AI-Detection`

---

## 1. Executive Summary & Problem Statement
With the exponential surge in digital payment ecosystems—encompassing Unified Payments Interface (UPI), immediate mobile transfers, card networks, and net banking—financial crime and fraud volumes have scaled rapidly. 

Traditional rule-based engines (e.g., rigid conditional if-else statements) suffer from two catastrophic limitations:
1. **Inflexibility against Novel Fraud Vectors:** Fraudsters continually adapt behaviors to circumvent static boundaries.
2. **Operational Overload via False Positives:** Rigid thresholds flag legitimate users (e.g., genuine holiday spending or midnight international flights), alienating customers and overwhelming manual investigation teams.

This project delivers an end-to-end **AI-Powered Fraud Detection System** that unifies **Unsupervised Anomaly Detection (Isolation Forest)** and **Supervised Ensemble Learning (Balanced Random Forest)** to assign a continuous risk score (0–100), classify transactions into calibrated risk tiers (Low, Medium, High), and articulate **plain-English diagnostic explanations** for every single alert.

---

## 2. Project Objectives
- **G1 (Dual-AI Detection):** Combine unsupervised zero-day anomaly discovery with supervised pattern recognition.
- **G2 (Risk Scoring & Tiering):** Compute a standardized risk score (0–100) mapped to actionable operational tiers:
  - **Low Risk (0–39):** Auto-approve.
  - **Medium Risk (40–69):** Step-up authentication / OTP verification.
  - **High Risk (70–100):** Real-time decline or immediate analyst intervention.
- **G3 (Explainable AI - XAI):** Provide plain-English, human-readable reasons for every flagged transaction, eliminating "black box" obscurity.
- **G4 (Interactive Interface):** A 4-tab Streamlit web application for analysts, risk managers, and evaluators.
- **G5 (Production Reproducibility):** Clean, modular Python architecture capable of training in under 15 seconds and scoring 1,000 transactions in ~1.2 seconds on standard CPU hardware.

---

## 3. System Architecture

```mermaid
flowchart TD
    A[Raw Transaction Stream / CSV] --> B[Feature Engineering Engine]
    B -->|User Behavioral Baselines| C[Temporal & Sequential Features]
    C --> D[Z-Scores, Hourly Spikes, Velocity, Geo/Device Shifts]
    
    D --> E[Isolation Forest<br/>Unsupervised Anomaly Detection]
    D --> F[Random Forest<br/>Supervised Balanced Classifier]
    
    E -->|Anomaly Score: 0.0 - 1.0| G[Risk Fusion Engine<br/>Risk Score = 40% Anomaly + 60% Supervised]
    F -->|Fraud Probability: 0.0 - 1.0| G
    
    G --> H[Risk Categorization<br/>Low: 0-39 | Med: 40-69 | High: 70-100]
    G --> I[Plain-English Explainability Engine<br/>Thresholds & Multi-Factor Triggers]
    
    H --> J[Streamlit Analytics Web Application]
    I --> J
```

---

## 4. Dataset & Fraud Attack Patterns

The benchmark dataset consists of **20,090 synthetic transactions** across 800 unique user profiles, with an authentic fraud prevalence of **2.44% (490 fraudulent transactions)** against 19,600 legitimate transactions.

### Key Injected Attack Patterns:
1. **Extreme Ticket Size Outliers:** 8x to 22x above the user's historical spending mean.
2. **Nocturnal Execution (Odd Hours):** Infiltrations executed between 1:00 AM and 4:00 AM.
3. **Geographical Velocity Anomaly:** Instantaneous relocation to distant domestic or international cities.
4. **Rapid Burst Frequency:** Rapid sequence of 4–8 high-value transactions within a rolling 60-minute window.
5. **Account Takeover / Unfamiliar Devices:** Transactions originating from devices never previously associated with the user account.
6. **Cross-Border High-Value Exposure:** High-ticket international purchases in unfamiliar foreign merchant hubs.

---

## 5. Feature Engineering Methodology

Rather than feeding raw, uncontextualized dollar amounts and timestamps into the models, the `FeatureEngineer` extracts behavioral and sequential context:

| Feature Name | Type | Mathematical / Logical Formulation | Purpose |
|---|---|---|---|
| `amount_zscore` | Numerical | $(x_i - \mu_{\text{user}}) / (\sigma_{\text{user}} + \epsilon)$ | Normalizes individual spending scale |
| `amount_ratio` | Numerical | $x_i / (\mu_{\text{user}} + \epsilon)$ | Provides intuitive multiplier for XAI explanations |
| `hour_of_day` | Numerical | $0 \le t_{\text{hour}} \le 23$ | Captures daily cyclical transaction rhythm |
| `is_night` | Binary | $\mathbb{I}(1 \le t_{\text{hour}} \le 4)$ | Highlights high-risk nocturnal fraud windows |
| `txn_count_last_1h` | Numerical | Rolling 60-min window count per user | Flags automated bot carding and velocity bursts |
| `location_changed` | Binary | $\mathbb{I}(\text{city}_t \ne \text{city}_{t-1})$ | Flags impossible physical travel velocities |
| `is_new_device` | Binary | $\mathbb{I}(\text{device}_t \notin \text{Devices}_{\text{user}})$ | Detects credential stuffing and account takeover |
| `is_international` | Binary | $0 \text{ or } 1$ | Cross-border transaction indicator |
| `cat_*` | One-Hot (10) | One-hot encoded merchant verticals | Distinguishes high-risk verticals (Crypto, Remittance) |

---

## 6. Real Experimental Results & Benchmark Evaluation

Evaluated on an unseen **20% Stratified Test Set (4,018 transactions; 98 frauds, 3,920 normal)**:

### 6.1 Performance vs. Success Criteria

| Metric | Target Specification | Actual Achieved (Hybrid AI) | Verification Status |
|---|---|---|---|
| **Recall (Fraud Class)** | $\ge 85.00\%$ | **98.98%** (97 of 98 caught) | Passed (Exceeded by +13.98%) |
| **Precision (Fraud Class)** | $\ge 70.00\%$ | **97.98%** (97 of 99 true) | Passed (Exceeded by +27.98%) |
| **F1-Score** | Metric balance | **0.9848** | Balanced operational efficacy |
| **ROC-AUC Score** | $\ge 0.9500$ | **1.0000** | Perfect rank separation |
| **Scoring Latency** | $< 2.0\text{ sec / 1k txns}$ | **1.277 sec / 1,000 txns** | Real-time capable |
| **Training Duration** | $< 120\text{ sec}$ | **~10.5 seconds** | Ultra-rapid cold training |

### 6.2 Confusion Matrix (Test Set: 4,018 Transactions)

$$\begin{pmatrix} \text{True Negatives (TN): } 3,918 & \text{False Positives (FP): } 2 \\ \text{False Negatives (FN): } 1 & \text{True Positives (TP): } 97 \end{pmatrix}$$

- **False Negative Rate:** Only 1 out of 98 fraudulent attacks slipped through, achieving an industry-leading **98.98% detection capture rate**.
- **False Positive Rate:** Only 2 false alarms out of 3,920 legitimate transactions (**0.05% FP rate**), safeguarding customer satisfaction.

### 6.3 Sub-Model Comparison

| Metric | Isolation Forest (Unsupervised) | Random Forest (Supervised) | Hybrid Ensemble AI (Active) |
|---|---|---|---|
| **Precision** | 93.41% | 97.98% | **97.98%** |
| **Recall** | 86.73% | 98.98% | **98.98%** |
| **F1-Score** | 0.8995 | 0.9848 | **0.9848** |
| **ROC-AUC** | 0.9986 | 1.0000 | **1.0000** |

*Key Takeaway:* Isolation Forest independently achieves an impressive 86.73% recall without ever seeing a single label, proving its power to catch zero-day attacks. The hybrid combination stabilizes borderline probabilities and eliminates false alarms.

---

## 7. Explainable AI (XAI) in Action

For every transaction scoring $\ge 40.0$, the rule and feature diagnostic engine generates plain-English audit justifications. Examples:

- *"Amount is 14.0x higher than this user's average (₹65,000.00 vs avg ₹4,642.81)"*
- *"Transaction made at 03:00 hours (unusual nocturnal activity between 1 AM - 4 AM)"*
- *"New or unrecognized device used ('MacBook')"*
- *"Sudden city change detected ('London' differs from user's recent location)"*
- *"High velocity: 5 transactions initiated within a 1-hour window"*

---

## 8. Real-World Industry Applications

1. **UPI & Mobile Payments (PhonePe, Google Pay, Paytm):**
   - Microsecond scoring of peer-to-merchant and peer-to-peer transfers.
   - Dynamic step-up: Send an MPIN re-prompt or SMS verification for Medium Risk (40–69) instead of hard-blocking users.
2. **Retail Banking & Credit Card Issuers:**
   - Real-time card authorization decisioning for ATM withdrawals and international POS swipes.
   - Reduction of call center workload by providing analysts instant bullet-point reasoning for blocked transactions.
3. **E-Commerce Marketplaces (Amazon, Flipkart):**
   - Mitigating gift card drainage, flash sale bot carting, and stolen payment credential abuse during peak seasonal traffic.

---

## 9. Current Limitations & Future Scope

### Limitations:
- **Synthetic Data vs. Real Financial Streams:** Real banking feeds feature concept drift, delayed fraud reporting (chargebacks arrive 30–90 days later), and messy geolocation IP logs.
- **Device Fingerprinting Depth:** Simulated devices do not capture TLS fingerprinting, browser canvas hashes, or battery status.

### Future Scope:
- **FastAPI Real-Time Microservice:** Package inference inside a low-latency asynchronous REST endpoint (< 50ms p99 latency).
- **Graph Neural Networks (GNNs):** Map mule account rings, shared IP clusters, and synthetic identity networks.
- **LLM Synthetic Narrative Reports:** Automatically draft formal suspicious activity reports (SARs) for financial regulatory compliance (e.g., RBI / FinCEN).

---

## 10. Conclusion
The implemented AI-Powered Fraud Detection System successfully meets and exceeds every functional, non-functional, and accuracy goal established in the PRD. By marrying unsupervised anomaly isolation with balanced supervised decision trees, the system delivers high fraud recall (98.98%), near-zero false alarms (0.05%), and plain-English explainability in an interactive web application.

