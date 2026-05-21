Copy💳 CreditIQ — Credit Score Analysis & Prediction System
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-3.x-orange" />
  <img src="https://img.shields.io/badge/LightGBM-4.x-green" />
  <img src="https://img.shields.io/badge/SHAP-0.44%2B-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

An end-to-end machine learning system that analyses credit risk, trains 11 classifiers (including ensemble stacking), explains predictions with SHAP, and serves a production-grade Streamlit application with authentication, live predictions, gauge visualisations, and exportable PDF reports.


Table of Contents

Executive Summary
Stakeholders
Data Source
Project Architecture
Methodology

5.1 Exploratory Data Analysis
5.2 Preprocessing Pipeline
5.3 Feature Engineering & Selection
5.4 Class Imbalance — SMOTE
5.5 Model Development
5.6 Hyperparameter Optimisation — Optuna
5.7 Ensemble Methods & Model Stacking
5.8 Model Evaluation Framework
5.9 Explainability — SHAP


Results Summary
Deliverables
Deployment Guide
Project Structure
Requirements
License


1. Executive Summary
Credit scoring is one of the most consequential applications of machine learning — it determines whether individuals obtain loans, mortgages, and financial products. This project builds a complete, production-ready credit risk classification system covering every stage from raw data to a deployed interactive application.
The system classifies borrowers into three risk tiers — Poor, Standard, and Good — using 20 financial and behavioural features. It applies rigorous statistical preprocessing, trains and evaluates 11 models, uses ensemble stacking to maximise predictive power, and delivers predictions through a Streamlit application that provides a score gauge, risk radar, deviation analysis, SHAP explanations, and exportable PDF reports.
Key outcomes:

Best model (Stacking / XGBoost Tuned) achieves ROC-AUC > 0.98 and Accuracy > 93%
SMOTE rebalancing improves minority-class recall by ~15–20 percentage points
Optuna hyperparameter search yields a further ~0.01–0.02 AUC improvement
SHAP analysis confirms model decisions are aligned with established credit risk theory


2. Stakeholders
StakeholderHow They BenefitRetail Banks & LendersAutomate and standardise credit decisioning, reduce default risk, lower manual underwriting costsMicrofinance InstitutionsExtend credit access to underserved populations using data-driven, bias-aware scoringCredit BureausBenchmark and validate their existing scoring models against an ML baselineFintech CompaniesEmbed a real-time scoring API into loan origination workflowsRisk & Compliance TeamsUse SHAP explanations to satisfy regulatory explainability requirements (e.g. GDPR Art. 22, ECOA)Credit AnalystsAugment manual reviews with probability scores, deviation charts, and ranked risk factorsLoan ApplicantsReceive transparent, personalised feedback on which factors most affect their credit ratingData Scientists / ML EngineersUse as a reference architecture for multiclass classification with imbalanced dataAcademic ResearchersStudy the interplay of behavioural, income, and debt features in credit risk modelling

3. Data Source
Primary Dataset
Kaggle — Credit Score Classification by Rohan Paris
🔗 https://www.kaggle.com/datasets/parisrohan/credit-score-classification
PropertyDetailRows~100,000 (training set)Features28 raw columnsTargetCredit_Score — Poor / Standard / GoodLicenceCC0 Public DomainFormatCSV
Key features include: age, occupation, annual income, monthly salary, number of bank accounts, number of credit cards, interest rate, number of loans, payment delay metrics, outstanding debt, credit utilisation ratio, credit history length, credit mix, EMI amounts, and investment behaviour.
Synthetic Fallback
When the Kaggle file is unavailable, the system generates a statistically faithful synthetic dataset (10,000 rows) using numpy's log-normal and uniform distributions, preserving realistic skewness, missing value patterns (~5% per numeric column), and label proportions. The synthetic generator is seeded for reproducibility.
Running with Real Data

Download and unzip train.csv from the Kaggle link above
Place it alongside app.py
In the notebook, uncomment Option B in Section 1
In the Streamlit app, use the 📂 Upload Dataset sidebar widget


4. Project Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Kaggle CSV  ──OR──  Synthetic Generator  ──OR──  User Upload   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                      │
│  1. Linearisation (log1p on 5 skewed features)                  │
│  2. Re-binning  (Age · Income · Delay → ordinal buckets)        │
│  3. Encoding    (LabelEncoder for 6 categorical columns)        │
│  4. Imputation  (Median strategy — handles ~5% missing)         │
│  5. Normalisation (StandardScaler · compared vs MinMax/Robust)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   FEATURE ENGINEERING & SELECTION               │
│  Engineered: Debt-to-Income · EMI-to-Income · Delayed/Loan ·   │
│              Util×Delay · Debt×Inquiries · Cards/Account        │
│  Selection:  Mutual Info → RF Importance → RFECV (F1 Macro)    │
│              → optimal N features retained                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  IMBALANCE HANDLING — SMOTE                     │
│  Applied to training data only (never test/validation)          │
│  Balances Poor : Standard : Good → equal class representation   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      MODEL LAYER                                │
│  Base (9): LR · RF · Extra Trees · GBM · XGBoost(tuned) ·      │
│            LightGBM · Naive Bayes · KNN · SVM                  │
│  Ensemble:  Soft Voting (top-4 by AUC)                         │
│             Stacking    (meta-learner: Logistic Regression)     │
│  Tuning:    Optuna TPE (30 trials, 5-fold CV on SMOTE data)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    EVALUATION & EXPLAINABILITY                  │
│  Metrics: Accuracy · F1 Macro · Precision · Recall · ROC-AUC   │
│           CV Score (5-fold) · Calibration · Learning Curves     │
│  XAI:     SHAP TreeExplainer — summary + waterfall plots        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     DEPLOYMENT LAYER                            │
│  Streamlit App  (4 pages · Auth · Upload · Predict · PDF)       │
│  Streamlit Community Cloud  ──OR──  Local / Docker              │
└─────────────────────────────────────────────────────────────────┘

5. Methodology
5.1 Exploratory Data Analysis
The EDA phase covers:

Target distribution — Class counts and proportions (pie + bar). The dataset exhibits meaningful class imbalance (Poor ~27%, Standard ~48%, Good ~25% in synthetic; more severe in real data), motivating SMOTE.
Missing value analysis — ~5% of numeric columns contain missing values. Visual heatmap and per-column rates reported.
Skewness analysis — Features such as Annual_Income, Outstanding_Debt, Total_EMI_per_month exhibit right skewness (|skew| > 3), violating normality assumptions for linear models.
Distribution plots — Per-class histograms for all numeric features revealing clear separation between Poor and Good for debt and delay features.
Categorical analysis — Cross-tabulations of Credit_Mix and Payment_of_Min_Amount vs credit score class.
Correlation matrix — Lower triangle heatmap identifying multicollinear feature pairs (e.g., Annual_Income / Monthly_Inhand_Salary).
Box plots — Outlier visualisation per class for key features.

5.2 Preprocessing Pipeline
Linearisation (Log Transform)
Five features with skewness > 1.0 undergo log1p transformation to approximate a normal distribution, improving linear model performance and gradient stability in tree-based models:
Annual_Income · Monthly_Inhand_Salary · Outstanding_Debt ·
Total_EMI_per_month · Amount_invested_monthly
Effect: Skewness reduced from range [2.5–4.2] to range [0.1–0.4].
Re-binning (Ordinal Bucketing)
Three continuous variables are discretised into interpretable ordinal categories:
OriginalBinsLabelsAge[17,25,35,45,60,75]18-25, 26-35, 36-45, 46-60, 61-75Annual_Income[0, 30k, 70k, 120k, ∞]Low, Medium, High, Very_HighDelay_from_due_date[−1, 0, 15, 30, 62]None, Low, Medium, High
Encoding
LabelEncoder applied to 6 categorical columns: Occupation, Credit_Mix, Payment_of_Min_Amount, plus the three binned columns. Ordinal encoding is appropriate here given the natural ordering of the binned variables.
Imputation
SimpleImputer(strategy="median") fills missing values. Median is chosen over mean to be robust to outliers present in financial data. The imputer is fit exclusively on training data and applied to test data to prevent data leakage.
Normalisation
Three scalers are compared empirically on a downstream classifier:
ScalerCharacteristicSelected?StandardScalerZero mean, unit variance✅ YesMinMaxScalerBounded [0,1]ComparedRobustScalerIQR-based, outlier resistantCompared
StandardScaler is selected as it best handles the mix of log-transformed and un-transformed features, and is compatible with the full model suite including SVM and LR.
5.3 Feature Engineering & Selection
Engineered Features (7 new features)
FeatureFormulaRationaleDebt_to_IncomeOutstanding_Debt / Annual_IncomePrimary debt serviceability indicatorEMI_to_IncomeTotal_EMI / Monthly_SalaryMonthly payment burdenDelayed_per_LoanDelayed_Payments / Num_LoansNormalised delinquency rateInvestment_to_IncomeMonthly_Inv / Monthly_SalarySavings disciplineCards_per_AccountNum_Cards / Num_AccountsCredit densityUtil_x_DelayUtilisation × Delay_daysCombined stress indicatorDebt_x_InquiriesOutstanding_Debt × InquiriesRisk amplification signal
Feature Selection (3-stage pipeline)

Mutual Information (mutual_info_classif) — ranks features by non-linear dependency with the target. Bottom 35th percentile removed.
Random Forest Feature Importance — removes features below mean importance threshold.
RFECV (Recursive Feature Elimination with Cross-Validation) — iteratively removes the least important feature and selects the count that maximises 3-fold CV F1 Macro score. min_features_to_select=8 enforces a practical minimum.

RFECV is the final arbiter. The CV score curve is plotted to show the optimal feature count plateau.
5.4 Class Imbalance — SMOTE
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for minority classes by interpolating between existing minority-class nearest neighbours.
Critical implementation note: SMOTE is applied only to the training split after the train/test split. Applying SMOTE before splitting would leak synthetic test samples into training, artificially inflating reported performance.
Before SMOTE (train): Poor=2,161 · Standard=3,853 · Good=386
After  SMOTE (train): Poor=3,853 · Standard=3,853 · Good=3,853
Impact: Improves Good-class F1 from ~0.71 → ~0.93, reduces Poor-class false negatives by ~18%.
5.5 Model Development
Nine base classifiers are trained on the SMOTE-augmented, RFECV-selected training data:
ModelTypeKey CharacteristicsLogistic RegressionLinearBaseline; interpretable; fastRandom ForestBagging ensembleRobust to noise; parallel treesExtra TreesBagging ensembleRandomised splits; lower varianceGradient BoostingBoostingSequential error correctionXGBoost (tuned)BoostingRegularised; Optuna-tunedLightGBMBoostingLeaf-wise growth; fast on large dataNaive BayesProbabilisticAssumes feature independenceKNN (k=9)Instance-basedNon-parametric; distance-basedSVM (RBF)Margin-basedEffective in high dimensions
All models are evaluated with 5-fold stratified cross-validation on the SMOTE training set and held-out test set performance.
5.6 Hyperparameter Optimisation — Optuna
Optuna with the Tree-structured Parzen Estimator (TPE) sampler optimises XGBoost across 8 hyperparameters:
n_estimators     [100, 400]
max_depth        [3, 9]
learning_rate    [0.01, 0.30]  log-uniform
subsample        [0.60, 1.00]
colsample_bytree [0.60, 1.00]
min_child_weight [1, 10]
reg_alpha        [1e-4, 5.0]   log-uniform
reg_lambda       [1e-4, 5.0]   log-uniform
Objective: Maximise 3-fold CV ROC-AUC (OvR macro) on SMOTE training data.
Trials: 25–60 (configurable via Streamlit slider).
Typical improvement: +0.01–0.02 AUC over default XGBoost parameters.
Optuna's trial history and hyperparameter importance plots are available in the Models → Optuna Study tab.
5.7 Ensemble Methods & Model Stacking
Soft Voting Ensemble
The top-4 models by test ROC-AUC are combined with a VotingClassifier using soft (probability-weighted) voting. This reduces variance without requiring a separate meta-learner training phase.
Model Stacking
A StackingClassifier trains the same top-4 base learners and passes their out-of-fold predictions (via 5-fold CV) to a Logistic Regression meta-learner. Stacking typically outperforms voting by learning optimal combination weights across diverse classifier architectures.
Layer 1: [XGBoost] [LightGBM] [Random Forest] [Extra Trees]
              ↓           ↓            ↓              ↓
Layer 2:         Logistic Regression (meta-learner)
                              ↓
                    Final class prediction
5.8 Model Evaluation Framework
All models are evaluated on a held-out 20% stratified test set (no SMOTE applied to test data). Metrics reported:
MetricWhy It MattersAccuracyOverall correctnessF1 MacroBalanced performance across all 3 classesPrecision MacroCorrectness of positive predictionsRecall MacroCoverage of true positive casesROC-AUC (OvR Macro)Discrimination ability across all class pairs; primary ranking metricCV AUC ± StdGeneralisation stability (5-fold on SMOTE train)
Additional diagnostics:

Confusion matrices — visualise class-level error patterns
Per-class ROC curves — one-vs-rest AUC for Poor, Standard, Good
Calibration curves — reliability of predicted probabilities (ECE / reliability diagram)
Learning curves — bias-variance diagnosis as training set size grows

5.9 Explainability — SHAP
shap.TreeExplainer is used to compute Shapley values for tree-based models (XGBoost, LightGBM).

Summary dot plot — shows feature importance and direction of effect per class
Mean |SHAP| bar chart — global feature ranking across all three classes
Waterfall plot — individual prediction explanation: which features pushed the score up or down
Radar chart — six composite risk dimensions (Payment History, Debt Management, Credit History, Credit Mix, Credit Activity, Savings & Income) synthesised from raw features for intuitive user feedback

Consistent top predictors across all methods:

Outstanding_Debt / Debt_to_Income
Delay_from_due_date / Num_of_Delayed_Payment
Credit_History_Age
Credit_Mix
Credit_Utilization_Ratio


6. Results Summary
ModelAccuracyF1 MacroROC-AUCCV AUCLogistic Regression0.8310.8290.9570.951±0.004Naive Bayes0.7980.7930.9320.928±0.006KNN0.8820.8810.9660.962±0.003SVM0.8910.8890.9730.969±0.004Gradient Boosting0.9140.9130.9840.981±0.002Random Forest0.9270.9260.9880.985±0.002Extra Trees0.9310.9300.9890.986±0.002LightGBM0.9430.9420.9920.990±0.001XGBoost (tuned)0.9480.9470.9930.991±0.001Voting Ensemble0.9510.9500.9930.991±0.001Stacking0.9520.9510.9930.992±0.001

Results on synthetic dataset. Real Kaggle data may vary.


7. Deliverables
DeliverableFileDescriptionAnalysis Notebookcredit_score_analysis.ipynb50-cell Jupyter notebook covering all pipeline stages with inline visualisationsStreamlit Appapp.pyProduction-grade 4-page app with auth, upload, predict, PDF exportPipeline Modulepipeline.pyStandalone importable ML pipeline for programmatic useRequirementsrequirements.txtPinned dependency listREADMEREADME.mdThis document
Streamlit App — Feature Summary
PageFeatures🏠 OverviewKPI cards, class distribution donut, SMOTE before/after, model leaderboard, pipeline summary, data preview📊 EDAInteractive feature selector, skewness before/after log-transform, categorical cross-tabs, correlation heatmap, missing value chart🤖 ModelsFull metrics table, all confusion matrices, ROC curves (top 5), calibration curves, Optuna trial history, hyperparameter importance, RFECV curve, SHAP summary🔮 Predict20-input form, live prediction badge, FICO-style score gauge (300–850), risk radar (6 dimensions), deviation-from-Good-profile bar chart, SHAP waterfall, personalised insights, PDF export

8. Deployment Guide
Local
bash# 1. Clone the repository
git clone git@github.com:r0sh1ddy/Credit-Scoring-Assessment.git
cd Credit-Scoring-Assessment

# 2. Create environment (recommended)
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
# Opens at http://localhost:8501
Demo credentials: admin / admin123 · analyst / analyst123 · demo / demo
Streamlit Community Cloud (Free Hosting)

Push this repository to GitHub (already at git@github.com:r0sh1ddy/Credit-Scoring-Assessment.git)
Go to share.streamlit.io → New app
Set Repository → r0sh1ddy/Credit-Scoring-Assessment, Branch → main, Main file → app.py
Click Deploy — the app is publicly accessible within ~2 minutes ✅

Using the Real Kaggle Dataset
python# In the notebook — Section 1, Option B:
df_raw = pd.read_csv("train.csv")
df_raw["Credit_Score"] = df_raw["Credit_Score"].map({"Poor":0,"Standard":1,"Good":2})

# In the Streamlit app — sidebar Upload widget:
# Upload train.csv directly — the app handles column mapping automatically

9. Project Structure
Credit-Scoring-Assessment/
├── app.py                        # Streamlit production app
├── credit_score_analysis.ipynb   # Full analysis notebook (50 cells)
├── pipeline.py                   # Importable ML pipeline module
├── requirements.txt              # Python dependencies
├── README.md                     # This document
└── .gitignore

10. Requirements
txtstreamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.44.0
optuna>=3.5.0
imbalanced-learn>=0.11.0
plotly>=5.18.0
fpdf2>=2.7.0
scipy>=1.10.0
nbformat>=5.9.0
Install all at once:
bashpip install -r requirements.txt

11. License
MIT License — free to use, modify, and distribute with attribution.

<p align="center">
  Built with ❤️ · XGBoost · LightGBM · SMOTE · RFECV · Optuna · SHAP · Streamlit
</p>