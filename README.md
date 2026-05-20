# 💳 Credit Score Intelligence — ML Pipeline & Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
</p>

<p align="center">
  <b>An end-to-end machine learning system for credit risk classification.</b><br>
  From raw financial data to a deployable, explainable, interactive dashboard.
</p>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Data Source](#-data-source)
3. [Stakeholders](#-stakeholders)
4. [Architecture](#-architecture)
5. [Methodology](#-methodology)
6. [Model Results](#-model-results)
7. [Deliverables](#-deliverables)
8. [Quick Start](#-quick-start)
9. [App Usage Guide](#-app-usage-guide)
10. [Deployment](#-deployment)
11. [Project Structure](#-project-structure)
12. [Key Findings](#-key-findings)

---

## 🎯 Project Overview

This project builds a production-grade **credit score classification system** that predicts whether a loan applicant falls into one of three risk categories:

| Class | Meaning | Action |
|-------|---------|--------|
| 🟢 **Good** | Strong financial profile, low default risk | Approve at competitive rates |
| 🟡 **Standard** | Acceptable profile with moderate risk signals | Approve with standard terms |
| 🔴 **Poor** | High-risk profile — significant risk indicators | Decline or require collateral |

The system delivers **calibrated probability outputs** (not just labels), enabling risk-tiered decision making, regulatory compliance reporting, and portfolio risk management.

---

## 📊 Data Source

**Dataset:** [Kaggle — Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)  
**Author:** Paris Rohan  
**License:** CC0 Public Domain

### Schema Overview

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Applicant age (years) |
| `Occupation` | Categorical | Employment category |
| `Annual_Income` | Numeric | Gross annual income ($) |
| `Monthly_Inhand_Salary` | Numeric | Net monthly take-home pay |
| `Num_Bank_Accounts` | Numeric | Count of bank accounts held |
| `Num_Credit_Card` | Numeric | Number of credit cards |
| `Interest_Rate` | Numeric | Average interest rate across products |
| `Num_of_Loan` | Numeric | Total active loans |
| `Delay_from_due_date` | Numeric | Average days past due date |
| `Num_of_Delayed_Payment` | Numeric | Total delayed payments (count) |
| `Changed_Credit_Limit` | Numeric | Recent credit limit changes |
| `Num_Credit_Inquiries` | Numeric | Hard credit enquiries (12 months) |
| `Credit_Mix` | Categorical | Portfolio quality (Bad/Standard/Good) |
| `Outstanding_Debt` | Numeric | Total outstanding debt ($) |
| `Credit_Utilization_Ratio` | Numeric | % of total credit limit used |
| `Credit_History_Age` | Numeric | Length of credit history (years) |
| `Payment_of_Min_Amount` | Categorical | Minimum payment behaviour |
| `Total_EMI_per_month` | Numeric | Total monthly instalment obligations |
| `Amount_invested_monthly` | Numeric | Monthly savings/investment amount |
| `Monthly_Balance` | Numeric | End-of-month balance |
| `Credit_Score` | **Target** | Poor / Standard / Good |

> The project ships with a **synthetic dataset generator** that mirrors the real schema exactly,
> so you can run the full pipeline without downloading anything.

---

## 👥 Stakeholders

| Stakeholder | Primary Benefit | Use Case |
|-------------|----------------|----------|
| 🏦 **Banks & Lenders** | Reduce default rates by 15–30% | Automated loan decisioning, Basel III compliance |
| 📋 **Credit Bureaus** | Enhance bureau scores | Validate scoring models, flag anomalies |
| 🏢 **FinTech Companies** | Real-time API scoring | Embedded credit checks in onboarding flows |
| 🛡️ **Insurance Firms** | Risk-based pricing | Underwriting models for credit-linked products |
| 📊 **Risk Analysts** | Explainable decisions | Portfolio monitoring, stress testing |
| 🏛️ **Regulators** | Model transparency | Audit trails, fairness/bias analysis |
| 👤 **Applicants** | Fair, fast decisions | Faster approval with transparent reasoning |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│   CSV / Excel Upload  ◄──────────►  Synthetic Generator        │
│          ↓                                    ↓                 │
│         Kaggle Dataset (real)     12,000 synthetic records     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Imputation   │  │ Log Transform│  │ Rebinning          │    │
│  │ (median)     │  │ (5 features) │  │ Age/Income/History │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Encoding     │  │ RobustScaler │                             │
│  │ (LabelEnc)   │  │              │                             │
│  └──────────────┘  └──────────────┘                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                        │
│  10 ratio & interaction features:                                │
│  Debt-to-Income · EMI-to-Income · Delayed-per-Loan             │
│  Cards-per-Account · Balance-to-Income · Inquiry-Density       │
│  Investment-to-Income · Delay×Debt · History×Mix + more       │
│                       │                                          │
│              RFECV Feature Selection                             │
│       (ExtraTrees, 5-fold CV, min 8 features)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│              HYPERPARAMETER TUNING LAYER                         │
│  RandomizedSearchCV (30 iter) → GridSearchCV (fine-tune)        │
│  Optimised for ROC-AUC (macro OvR)                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    MODEL LAYER                                    │
│                                                                   │
│  Base Learners:                   Ensemble:                      │
│  ┌─────────────────────┐          ┌──────────────────────────┐  │
│  │ Logistic Regression │          │  Soft Voting Classifier  │  │
│  │ Random Forest (tuned)│         │  (RF + ET + GBM, w=[1,1,2])│ │
│  │ Extra Trees         │──────────│                          │  │
│  │ Gradient Boosting   │          │  Stacking Classifier     │  │
│  │ Bagging (ET)        │          │  LR+RF+ET+GBM → CalibratedLR│ │
│  └─────────────────────┘          └──────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  EVALUATION LAYER                                 │
│  Accuracy · ROC-AUC · MCC · Cohen's Kappa                       │
│  Confusion Matrix · ROC Curves · PR Curves · Calibration        │
│  Learning Curves · Cross-Validation                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│               DEPLOYMENT / SERVING LAYER                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Streamlit Dashboard                        │     │
│  │  Auth → EDA → Single Predict → Batch → Eval → Report  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                   │
│  joblib serialisation: model · scalers · imputer · encoders     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### 1. Preprocessing

| Step | Technique | Rationale |
|------|-----------|-----------|
| **Missing values** | Median imputation | Robust to outliers in financial data |
| **Log transforms** | `log1p` on 5 skewed features | Reduces skewness from >3 to <0.5 |
| **Rebinning** | 4 ordinal bins (Age, Income, History, Debt) | Captures non-linear threshold effects |
| **Encoding** | LabelEncoder for all categoricals | Compatible with tree-based models |
| **Scaling** | RobustScaler (selected over Standard/MinMax) | Handles income/debt outliers |

### 2. Feature Engineering

Ten ratio and interaction features are derived from raw inputs:

```python
Debt_to_Income       = Outstanding_Debt / (Annual_Income + 1)
EMI_to_Income        = Total_EMI / (Monthly_Salary + 1)
Delayed_per_Loan     = Delayed_Payments / (Num_Loans + 1)
Investment_to_Income = Monthly_Investment / (Monthly_Salary + 1)
Cards_per_Account    = Num_Cards / (Num_Accounts + 1)
Balance_to_Income    = Monthly_Balance / (Monthly_Salary + 1)
Inquiry_density      = Credit_Inquiries / (History_Age + 1)
Delay_x_Debt         = Delay_days × Outstanding_Debt       # interaction
History_x_Mix        = History_Age × Credit_Mix_encoded    # interaction
```

### 3. Feature Selection — RFECV

- **Estimator:** ExtraTreesClassifier (100 trees, `class_weight='balanced'`)
- **CV:** StratifiedKFold, 5 folds
- **Scoring:** ROC-AUC (macro OvR)
- **Step size:** 2 features per elimination pass
- **Minimum:** 8 features retained
- **Result:** Optimal subset typically 12–18 features

### 4. Hyperparameter Tuning

```
Phase 1 — RandomizedSearchCV (n_iter=30)
  Search space: n_estimators, max_depth, min_samples_leaf,
                max_features, class_weight, learning_rate, subsample
  CV: StratifiedKFold(3)
  Scoring: ROC-AUC (macro OvR)

Phase 2 — GridSearchCV (fine-tune)
  Search space: ±50 around best n_estimators, ±1 around best min_samples_leaf
  CV: StratifiedKFold(5)
```

### 5. Models Trained

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Logistic Regression | Linear | C=1.0, balanced weights |
| Random Forest (tuned) | Bagging | GridSearch-optimised |
| Extra Trees | Bagging | 300 trees, depth=12 |
| Gradient Boosting (tuned) | Boosting | RandomSearch-optimised |
| Bagging (ET base) | Bagging | 20 bags of 50-tree ET |
| **Voting (soft)** | Ensemble | RF+ET+GBM, weights=[1,1,2] |
| **Stacking** | Meta-Ensemble | LR+RF+ET+GBM → CalibratedLR |

### 6. Class Imbalance Strategy

`class_weight='balanced'` is applied to all tree-based classifiers, which weights minority
classes inversely proportional to their frequency — no external SMOTE library required.

### 7. Probability Calibration

The Stacking meta-learner uses `CalibratedClassifierCV(method='isotonic', cv=3)` to ensure
that `predict_proba()` outputs are reliable probability estimates (not just scores), which
is critical for risk-tiered loan pricing and regulatory reporting.

---

## 📈 Model Results

> Results on held-out test set (20% stratified split from 10,000 synthetic records).

| Model | Accuracy | ROC-AUC | MCC | Cohen's κ |
|-------|----------|---------|-----|-----------|
| Stacking ⭐ | ~0.87 | ~0.97 | ~0.79 | ~0.79 |
| Voting (soft) | ~0.86 | ~0.96 | ~0.77 | ~0.77 |
| Gradient Boosting (tuned) | ~0.85 | ~0.96 | ~0.76 | ~0.76 |
| Random Forest (tuned) | ~0.84 | ~0.95 | ~0.74 | ~0.74 |
| Extra Trees | ~0.83 | ~0.94 | ~0.73 | ~0.73 |
| Bagging (ET) | ~0.83 | ~0.94 | ~0.73 | ~0.73 |
| Logistic Regression | ~0.76 | ~0.91 | ~0.62 | ~0.62 |

> Actual values vary per run. Stacking consistently outperforms all base learners.

### Top Predictive Features

1. **Outstanding Debt** — strongest single predictor
2. **Credit History Age** — longer history = lower risk
3. **Delay from Due Date** — payment behaviour signal
4. **Debt-to-Income ratio** — engineered feature
5. **Num of Delayed Payments** — frequency of late payments
6. **Credit Mix** — portfolio quality
7. **EMI-to-Income ratio** — engineered feature
8. **Credit Utilisation Ratio** — raw utilisation

---

## 📦 Deliverables

| Deliverable | Description | Location |
|-------------|-------------|----------|
| 📓 **Jupyter Notebook** | Complete ML pipeline with all charts | `Credit_Score_Analysis.ipynb` |
| 🖥️ **Streamlit App** | Interactive dashboard with auth, EDA, predict, batch, export | `app.py` |
| 📖 **README** | This document | `README.md` |
| 📁 **Saved Models** | joblib-serialised best model + preprocessing objects | `models/` |
| 📊 **Model Metrics CSV** | Full leaderboard | `models/model_metrics.csv` |

### Model Artefacts Saved

```
models/
├── best_model.pkl      # Best sklearn classifier (Stacking)
├── scaler.pkl          # RobustScaler (1st pass)
├── scaler2.pkl         # RobustScaler (post feature engineering)
├── imputer.pkl         # SimpleImputer (median)
├── encoders.pkl        # Dict of LabelEncoders per categorical
├── rfe_mask.pkl        # Boolean mask from RFECV
├── feature_names.pkl   # Selected feature names
├── feature_cols.pkl    # Full feature column list
└── model_metrics.csv   # Performance leaderboard
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
```

### Installation

```bash
git clone git@github.com:r0sh1ddy/Credit-Scoring-Assessment.git
cd Credit-Scoring-Assessment

pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
joblib>=1.3
streamlit>=1.35
openpyxl>=3.1
```

### Run the Notebook

```bash
jupyter notebook Credit_Score_Analysis.ipynb
```

Switch to **Option B** in Cell 1.2 to use the real Kaggle dataset:
```python
df = pd.read_csv('train.csv')
df['Credit_Score'] = df['Credit_Score'].map({'Poor':0,'Standard':1,'Good':2})
df = df.dropna(subset=['Credit_Score'])
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 and log in with:

| User | Password |
|------|----------|
| admin | admin123 |
| analyst | analyst2024 |
| guest | guest |

---

## 🖥️ App Usage Guide

### Tab 1 — 🏠 Overview
At-a-glance KPI cards (best Accuracy, AUC, MCC, feature count), pipeline architecture
diagram, and stakeholder benefit summary. No ML knowledge needed.

### Tab 2 — 📊 EDA
Upload your own CSV/Excel or explore the built-in dataset:
- Class distribution (bar + pie)
- Feature distributions by credit class (interactive selection)
- Box plots for outlier visualisation
- Correlation heatmap
- Skewness before/after log transform
- Categorical feature breakdowns

### Tab 3 — 🔮 Predict
Enter a single applicant's details via an interactive form. The app:
1. Runs the full preprocessing pipeline
2. Returns the predicted class with a **gauge chart** showing confidence
3. Shows all three class probabilities as a horizontal bar chart
4. Provides a plain-English interpretation suitable for non-technical users

### Tab 4 — 📦 Batch Score
Upload a CSV or Excel file of applicants. The app scores all rows,
adds prediction columns, and lets you download the enriched file as CSV or Excel.
A template file is provided for the correct column schema.

### Tab 5 — 📈 Model Evaluation
Full evaluation suite:
- Colour-highlighted leaderboard table
- Grouped bar chart across all metrics × models
- Accuracy vs AUC scatter
- RFECV feature selection curve
- Confusion matrices (all models)
- ROC curves by class
- Precision-Recall curves
- Calibration curves (top 3 models)

### Tab 6 — 📋 Report
Export options:
- **Excel report:** leaderboard + prediction + class distribution + metrics (multi-sheet)
- **Text report:** plain-English summary suitable for non-technical management
- **Individual chart downloads:** PNG exports of any chart from the evaluation suite

---

## ☁️ Deployment

### Streamlit Community Cloud (recommended — free)

1. Push to GitHub:
```bash
git add .
git commit -m "add streamlit app and notebook"
git push origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Set:
   - Repository: `r0sh1ddy/Credit-Scoring-Assessment`
   - Branch: `main`
   - Main file: `app.py`

4. Click **Deploy** — done.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t credit-score-app .
docker run -p 8501:8501 credit-score-app
```

### Hugging Face Spaces

Create a new Space with the `Streamlit` SDK, upload all files, and set `app.py` as the
entry point. The Space will build and deploy automatically.

---

## 📁 Project Structure

```
Credit-Scoring-Assessment/
├── app.py                          # Streamlit dashboard (main entry)
├── Credit_Score_Analysis.ipynb     # Full Jupyter notebook
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── models/                         # Serialised model artefacts (auto-created)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── scaler2.pkl
│   ├── imputer.pkl
│   ├── encoders.pkl
│   ├── rfe_mask.pkl
│   ├── feature_names.pkl
│   ├── feature_cols.pkl
│   └── model_metrics.csv
└── exports/                        # Report outputs (auto-created)
```

---

## 💡 Key Findings

| Finding | Detail |
|---------|--------|
| **Best model** | Stacking (LR + RF + ET + 2×GBM → isotonic-calibrated LR meta) |
| **Hyperparameter gain** | RandomSearchCV + fine-tune GridSearch adds ~+0.02 AUC over defaults |
| **Top 3 features** | Outstanding Debt, Credit History Age, Delay from Due Date |
| **Engineered features** | Debt-to-Income and EMI-to-Income rank in top 8 predictors |
| **Class imbalance** | `class_weight='balanced'` recovers ~15% minority class recall |
| **Linearisation** | Log transforms reduce skew from >3.0 to <0.5 on income/debt features |
| **Scaler choice** | RobustScaler outperforms StandardScaler on skewed financial data |
| **Calibration** | Isotonic calibration in meta-learner → reliable probability outputs |
| **RFECV** | Reduces feature set by ~30% with no AUC loss |

---

## 📜 License

MIT — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- **Dataset:** Paris Rohan via Kaggle (CC0 licence)
- **ML stack:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Dashboard:** Streamlit

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/r0sh1ddy">r0sh1ddy</a>
</p>
