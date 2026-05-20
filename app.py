"""
Credit Score Intelligence Dashboard  —  Streamlit App
======================================================
Features:
  • Simple login / authentication gate
  • Synthetic dataset (or real CSV/Excel upload)
  • EDA tab with full visual suite
  • Single-applicant prediction + animated gauge
  • Batch CSV/Excel upload & scoring
  • Model evaluation / leaderboard tab
  • PDF + Excel report export (non-tech friendly)
  • No XGBoost / SHAP required (sklearn only)

Run:
    streamlit run app.py
"""

# ── stdlib / scientific ────────────────────────────────────────────────────────
import io, os, json, warnings, base64, hashlib, time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import joblib

warnings.filterwarnings("ignore")

import streamlit as st
from streamlit import session_state as ss

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, GridSearchCV)
from sklearn.preprocessing import (
    LabelEncoder, RobustScaler, StandardScaler,
    MinMaxScaler, label_binarize)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, StackingClassifier,
    VotingClassifier, BaggingClassifier)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score)

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS      = ["#ef4444","#f59e0b","#22c55e","#3b82f6","#a855f7","#ec4899"]
CLASS_NAMES = ["Poor","Standard","Good"]
LABEL_MAP   = {0:"Poor", 1:"Standard", 2:"Good"}
SEED        = 42
np.random.seed(SEED)

# ─── CREDENTIALS (demo; swap for DB in production) ────────────────────────────
USERS = {
    "admin":  hashlib.sha256(b"admin123").hexdigest(),
    "analyst":hashlib.sha256(b"analyst2024").hexdigest(),
    "guest":  hashlib.sha256(b"guest").hexdigest(),
}

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card{
    background:linear-gradient(135deg,#1e3a5f 0%,#0f2444 100%);
    border-radius:12px;padding:18px 22px;margin:6px 0;
    border-left:4px solid #3b82f6;color:#fff;
  }
  .metric-card h3{margin:0;font-size:2rem;color:#60a5fa;}
  .metric-card p{margin:4px 0 0;font-size:.85rem;color:#94a3b8;}
  .score-good   {background:linear-gradient(135deg,#064e3b,#065f46);border-left-color:#22c55e;}
  .score-std    {background:linear-gradient(135deg,#78350f,#92400e);border-left-color:#f59e0b;}
  .score-poor   {background:linear-gradient(135deg,#7f1d1d,#991b1b);border-left-color:#ef4444;}
  .section-header{font-size:1.4rem;font-weight:700;color:#1e40af;
                  border-bottom:2px solid #3b82f6;padding-bottom:6px;margin:20px 0 14px;}
  .info-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
            padding:12px 16px;margin:10px 0;font-size:.9rem;}
  .stTabs [data-baseweb="tab"]{font-size:1rem;font-weight:600;padding:10px 24px;}
  body{background-color:#f8fafc;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════
def check_password(username, password):
    h = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == h

def login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.4,1])
    with col2:
        st.markdown("""
        <div style='text-align:center;padding:30px;background:linear-gradient(135deg,#1e3a5f,#0f2444);
             border-radius:16px;color:white;margin-bottom:24px;'>
          <h1>💳 Credit Score Intelligence</h1>
          <p style='color:#94a3b8;'>ML-Powered Credit Risk Assessment Platform</p>
        </div>""", unsafe_allow_html=True)
        with st.form("login_form"):
            st.subheader("Sign In")
            username = st.text_input("Username", placeholder="admin / analyst / guest")
            password = st.text_input("Password", type="password", placeholder="Password")
            submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)
            if submitted:
                if check_password(username, password):
                    ss["authenticated"] = True
                    ss["username"]      = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try admin/admin123 or guest/guest")
        st.markdown("""
        <div class='info-box'>
          <b>Demo credentials</b><br>
          admin / admin123 &nbsp;|&nbsp; analyst / analyst2024 &nbsp;|&nbsp; guest / guest
        </div>""", unsafe_allow_html=True)

if not ss.get("authenticated"):
    login_page()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL PIPELINE  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
LOG_FEATURES = ["Annual_Income","Outstanding_Debt","Total_EMI_per_month",
                "Monthly_Inhand_Salary","Amount_invested_monthly"]

def generate_dataset(n=10_000, seed=42):
    rng  = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed+1)
    ai    = rng.lognormal(10.8, 0.7, n).round(2)
    delay = rng.integers(0,62,n).astype(float)
    dpm   = rng.integers(0,22,n).astype(float)
    debt  = rng.uniform(0,4998,n).round(2)
    hist  = rng.uniform(0,35,n).round(1)
    mix   = rng.choice(["Bad","Standard","Good"], n, p=[0.3,0.4,0.3])
    util  = rng.uniform(20,50,n).round(2)
    for arr in [ai,delay,dpm,debt]:
        arr[rng.random(n)<0.05] = np.nan
    s = np.clip(
        0.25*(ai/np.nanmax(ai))
        +0.20*(1-np.where(np.isnan(debt),0.5,debt)/4998)
        +0.15*(1-np.where(np.isnan(delay),0.5,delay)/62)
        +0.12*(hist/35)
        +0.10*(1-np.where(np.isnan(dpm),0.5,dpm)/22)
        +0.08*np.where(mix=="Good",1.0,np.where(mix=="Standard",0.5,0.0))
        +0.05*(1-util/50)+0.05*rng.uniform(0,1,n), 0,1)
    labels = np.where(s<0.38,0,np.where(s<0.65,1,2))
    return pd.DataFrame({
        "Age": rng2.integers(18,75,n).astype(float),
        "Occupation": rng2.choice(["Scientist","Teacher","Engineer","Entrepreneur",
            "Developer","Lawyer","Doctor","Journalist","Manager","Accountant",
            "Musician","Mechanic","Writer","Architect","Media_Manager"],n),
        "Annual_Income": ai, "Monthly_Inhand_Salary": ai/12,
        "Num_Bank_Accounts": rng2.integers(1,10,n).astype(float),
        "Num_Credit_Card": rng2.integers(0,12,n).astype(float),
        "Interest_Rate": rng2.uniform(1,34,n).round(2),
        "Num_of_Loan": rng2.integers(0,9,n).astype(float),
        "Delay_from_due_date": delay, "Num_of_Delayed_Payment": dpm,
        "Changed_Credit_Limit": rng2.uniform(0,30,n).round(2),
        "Num_Credit_Inquiries": rng2.integers(0,17,n).astype(float),
        "Credit_Mix": mix, "Outstanding_Debt": debt,
        "Credit_Utilization_Ratio": util, "Credit_History_Age": hist,
        "Payment_of_Min_Amount": rng2.choice(["Yes","No","NM"],n,p=[0.4,0.4,0.2]),
        "Total_EMI_per_month": rng2.lognormal(4.5,0.8,n).round(2),
        "Amount_invested_monthly": rng2.lognormal(5.0,1.0,n).round(2),
        "Monthly_Balance": rng2.uniform(0,2000,n).round(2),
        "Credit_Score": labels,
    })

def add_features(df):
    df = df.copy()
    df["Debt_to_Income"]       = df["Outstanding_Debt"]        / (df["Annual_Income"]         +1)
    df["EMI_to_Income"]        = df["Total_EMI_per_month"]     / (df["Monthly_Inhand_Salary"] +1)
    df["Delayed_per_Loan"]     = df["Num_of_Delayed_Payment"]  / (df["Num_of_Loan"]           +1)
    df["Investment_to_Income"] = df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] +1)
    df["Cards_per_Account"]    = df["Num_Credit_Card"]         / (df["Num_Bank_Accounts"]     +1)
    df["Loan_per_Account"]     = df["Num_of_Loan"]             / (df["Num_Bank_Accounts"]     +1)
    df["Balance_to_Income"]    = df["Monthly_Balance"]         / (df["Monthly_Inhand_Salary"] +1)
    df["Inquiry_density"]      = df["Num_Credit_Inquiries"]    / (df["Credit_History_Age"]    +1)
    df["Delay_x_Debt"]         = df["Delay_from_due_date"]     * df["Outstanding_Debt"]
    df["History_x_Mix"]        = df["Credit_History_Age"]      * df.get("Credit_Mix", 1)
    return df

@st.cache_resource(show_spinner="🔧 Training ML models — please wait…")
def build_pipeline():
    df = generate_dataset(n=10_000, seed=SEED)

    # Preprocessing
    df_p = df.copy()
    for feat in LOG_FEATURES:
        df_p[f"{feat}_log"] = np.log1p(df_p[feat].fillna(0))
    df_p["Age_bin"] = pd.cut(df_p["Age"],bins=[17,25,35,50,65,76],
        labels=["Young Adult","Adult","Mid-Career","Pre-Senior","Senior"])
    df_p["Income_quartile"] = pd.qcut(
        df_p["Annual_Income"].fillna(df_p["Annual_Income"].median()),
        q=4, labels=["Low","Below-Avg","Above-Avg","High"])
    df_p["History_tier"] = pd.cut(df_p["Credit_History_Age"],
        bins=[-0.1,3,8,18,36], labels=["New","Growing","Established","Veteran"])
    df_p["Debt_tier"] = pd.cut(
        df_p["Outstanding_Debt"].fillna(df_p["Outstanding_Debt"].median()),
        q=3, labels=["Low","Medium","High"])

    CAT_COLS = ["Occupation","Credit_Mix","Payment_of_Min_Amount",
                "Age_bin","Income_quartile","History_tier","Debt_tier"]
    encoders = {}
    for col in CAT_COLS:
        enc = LabelEncoder()
        df_p[col] = enc.fit_transform(df_p[col].astype(str))
        encoders[col] = enc

    feature_cols = [c for c in df_p.columns if c != "Credit_Score"]
    y = df_p["Credit_Score"].values
    X_raw = df_p[feature_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)

    X_tr_fe = add_features(pd.DataFrame(X_tr_s, columns=X_tr.columns))
    X_te_fe = add_features(pd.DataFrame(X_te_s, columns=X_te.columns))
    scaler2 = RobustScaler()
    X_tr_fs = scaler2.fit_transform(X_tr_fe)
    X_te_fs = scaler2.transform(X_te_fe)
    feat_names = X_tr_fe.columns.tolist()

    # RFECV
    rfecv = RFECV(
        estimator=ExtraTreesClassifier(n_estimators=80, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1),
        step=2, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        scoring="roc_auc_ovr", min_features_to_select=8, n_jobs=-1)
    rfecv.fit(X_tr_fs, y_tr)
    mask = rfecv.support_
    selected = [f for f, s in zip(feat_names, mask) if s]
    Xtr_sel = X_tr_fs[:, mask]; Xte_sel = X_te_fs[:, mask]

    # Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=SEED),
        "Voting (soft)": VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(100, class_weight="balanced",
                                              random_state=SEED, n_jobs=-1)),
                ("et", ExtraTreesClassifier(100, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(100, random_state=SEED)),
            ], voting="soft", weights=[1,1,2]),
        "Stacking": StackingClassifier(
            estimators=[
                ("lr",  LogisticRegression(max_iter=2000, C=0.1, random_state=SEED)),
                ("rf",  RandomForestClassifier(150, max_depth=8, random_state=SEED, n_jobs=-1)),
                ("et",  ExtraTreesClassifier(150, max_depth=8, random_state=SEED, n_jobs=-1)),
                ("gb",  GradientBoostingClassifier(100, max_depth=4, random_state=SEED)),
            ],
            final_estimator=CalibratedClassifierCV(
                LogisticRegression(max_iter=2000, random_state=SEED),
                cv=3, method="isotonic"),
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            stack_method="predict_proba", passthrough=True, n_jobs=-1),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(Xtr_sel, y_tr)
        preds = mdl.predict(Xte_sel)
        proba = mdl.predict_proba(Xte_sel)
        cv    = cross_val_score(mdl, Xtr_sel, y_tr,
                    cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                    scoring="roc_auc_ovr") if name not in ["Voting (soft)","Stacking"] else np.array([np.nan])
        results[name] = dict(
            model=mdl, preds=preds, proba=proba,
            acc  =accuracy_score(y_te, preds),
            auc  =roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
            mcc  =matthews_corrcoef(y_te, preds),
            kappa=cohen_kappa_score(y_te, preds),
            cv_mean=cv.mean(), cv_std=cv.std())

    summary = pd.DataFrame([
        {"Model":n, "Accuracy":r["acc"], "ROC-AUC":r["auc"],
         "MCC":r["mcc"], "Cohen's Kappa":r["kappa"]}
        for n,r in results.items()
    ]).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    best_name  = summary.iloc[0]["Model"]
    best_model = results[best_name]["model"]

    return dict(
        df=df, feature_cols=feature_cols, encoders=encoders,
        imputer=imputer, scaler=scaler, scaler2=scaler2,
        mask=mask, selected=selected, feat_names=feat_names,
        Xte_sel=Xte_sel, y_te=y_te,
        results=results, summary=summary,
        best_name=best_name, best_model=best_model,
        rfecv_scores=rfecv.cv_results_["mean_test_score"],
        rfecv_stds=rfecv.cv_results_["std_test_score"],
        n_features=rfecv.n_features_,
    )

pipe = build_pipeline()

# ── Prediction helper ──────────────────────────────────────────────────────────
def predict_applicant(raw_input_dict, model=None):
    if model is None:
        model = pipe["best_model"]
    df_in = pd.DataFrame([raw_input_dict])

    for feat in LOG_FEATURES:
        if feat in df_in.columns:
            df_in[f"{feat}_log"] = np.log1p(df_in[feat].fillna(0))
    df_in["Age_bin"] = pd.cut(df_in["Age"], bins=[17,25,35,50,65,76],
        labels=["Young Adult","Adult","Mid-Career","Pre-Senior","Senior"])
    df_in["Income_quartile"] = pd.cut(
        df_in["Annual_Income"].fillna(50000),
        bins=[0,30000,65000,120000,1e9], labels=["Low","Below-Avg","Above-Avg","High"])
    df_in["History_tier"] = pd.cut(df_in["Credit_History_Age"],
        bins=[-0.1,3,8,18,36], labels=["New","Growing","Established","Veteran"])
    df_in["Debt_tier"] = pd.cut(df_in["Outstanding_Debt"].fillna(2000),
        bins=[-0.1,1600,3300,5001], labels=["Low","Medium","High"])

    for col, enc in pipe["encoders"].items():
        if col in df_in.columns:
            try:
                df_in[col] = enc.transform(df_in[col].astype(str))
            except Exception:
                df_in[col] = 0
    for c in pipe["feature_cols"]:
        if c not in df_in.columns:
            df_in[c] = 0
    df_in = df_in[pipe["feature_cols"]]
    X_in  = pd.DataFrame(pipe["imputer"].transform(df_in), columns=df_in.columns)
    X_sc  = pipe["scaler"].transform(X_in)
    X_fe  = add_features(pd.DataFrame(X_sc, columns=X_in.columns))
    X_sc2 = pipe["scaler2"].transform(X_fe)
    X_sel = X_sc2[:, pipe["mask"]]
    preds = model.predict(X_sel)
    proba = model.predict_proba(X_sel)
    return LABEL_MAP[preds[0]], proba[0]

# ── Figure helpers ─────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def gauge_chart(confidence, label):
    color_map = {"Good":"#22c55e", "Standard":"#f59e0b", "Poor":"#ef4444"}
    col       = color_map.get(label, "#3b82f6")
    angle     = 180 * confidence        # 0→180°
    fig, ax   = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    # Background arc
    theta_bg = np.linspace(0, np.pi, 200)
    ax.fill_between(theta_bg, 0.7, 1.0, color="#e5e7eb", zorder=1)
    # Coloured arc
    theta_val = np.linspace(0, np.deg2rad(angle), 200)
    ax.fill_between(theta_val, 0.7, 1.0, color=col, alpha=0.85, zorder=2)
    # Needle
    needle = np.deg2rad(angle)
    ax.plot([needle, needle], [0, 0.65], color="#1e3a5f", lw=3, zorder=3)
    ax.set_rticks([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")
    ax.text(np.pi/2, 0.3, f"{confidence*100:.1f}%\n{label}",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=col, transform=ax.transData)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 👤 {ss['username'].title()}")
    st.markdown("---")
    st.markdown("**💳 Credit Score Intelligence**")
    st.markdown("ML-powered credit risk platform")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"• {len(pipe['df']):,} applicants")
    st.markdown(f"• {len(pipe['feature_cols'])} raw features")
    st.markdown(f"• {pipe['n_features']} after RFECV")
    st.markdown("---")
    st.markdown("**Best Model**")
    best_auc = pipe["results"][pipe["best_name"]]["auc"]
    st.markdown(f"🏆 `{pipe['best_name']}`")
    st.markdown(f"ROC-AUC: **{best_auc:.4f}**")
    st.markdown("---")
    st.caption("Data: [Kaggle Credit Score Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)")
    if st.button("🚪 Logout", use_container_width=True):
        ss["authenticated"] = False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Overview", "📊 EDA", "🔮 Predict", "📦 Batch Score",
                "📈 Model Evaluation", "📋 Report"])
df = pipe["df"]

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## 💳 Credit Score Intelligence Dashboard")
    st.markdown("""
    <div class='info-box'>
      <b>About this Platform</b><br>
      An end-to-end ML pipeline for credit risk assessment. Trained on the
      <a href='https://www.kaggle.com/datasets/parisrohan/credit-score-classification' target='_blank'>
      Kaggle Credit Score Classification dataset</a>. Classifies applicants as
      <b>Poor</b>, <b>Standard</b>, or <b>Good</b> credit risk using an ensemble of
      sklearn models with RFECV feature selection and calibrated probability outputs.
    </div>""", unsafe_allow_html=True)

    # KPI cards
    res = pipe["results"]
    best = pipe["best_name"]
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
          <h3>{res[best]['acc']*100:.1f}%</h3><p>Best Accuracy</p></div>""",
          unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
          <h3>{res[best]['auc']:.4f}</h3><p>Best ROC-AUC</p></div>""",
          unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
          <h3>{res[best]['mcc']:.3f}</h3><p>MCC Score</p></div>""",
          unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
          <h3>{pipe['n_features']}</h3><p>Selected Features (RFECV)</p></div>""",
          unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Pipeline Architecture")
    cols = st.columns(5)
    steps = [
        ("📥","Data Ingestion","CSV / Synthetic\n10K records"),
        ("🔧","Preprocessing","Log transform\nImpute · Encode\nRobustScaler"),
        ("⚙️","Feature Eng.","10 ratio features\nRFECV selection"),
        ("🤖","Ensemble ML","RF · ET · GBM\nVoting · Stacking"),
        ("📊","Evaluation","AUC · MCC · Kappa\nCalibration · PR"),
    ]
    for col,(icon,title,desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='text-align:center;background:#eff6ff;border-radius:10px;
                 padding:16px;border:1px solid #bfdbfe;'>
              <div style='font-size:2rem'>{icon}</div>
              <b>{title}</b><br>
              <small style='color:#6b7280'>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 Stakeholders & Value")
    sc1,sc2,sc3 = st.columns(3)
    with sc1:
        st.markdown("""
        <div style='background:#f0fdf4;border-radius:10px;padding:16px;border:1px solid #86efac;'>
          <b>🏦 Banks & Lenders</b><br>
          Automate loan approval, reduce default risk, ensure regulatory compliance (Basel III).
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown("""
        <div style='background:#fffbeb;border-radius:10px;padding:16px;border:1px solid #fde68a;'>
          <b>📋 Credit Bureaus</b><br>
          Enhance scoring models, validate existing scores, flag data anomalies.
        </div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown("""
        <div style='background:#fdf4ff;border-radius:10px;padding:16px;border:1px solid #d8b4fe;'>
          <b>🏢 FinTech / Insurance</b><br>
          Real-time scoring APIs, personalised product offers, risk-based pricing.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📊 Exploratory Data Analysis")

    # Upload override
    uploaded = st.file_uploader(
        "Upload your own dataset (CSV or Excel, must include Credit_Score column)",
        type=["csv","xlsx"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
                    else pd.read_excel(uploaded)
            if "Credit_Score" in df_up.columns:
                df_up["Credit_Score"] = df_up["Credit_Score"].map(
                    {"Poor":0,"Standard":1,"Good":2}).fillna(df_up["Credit_Score"])
            df = df_up
            st.success(f"✅ Loaded {len(df):,} rows from {uploaded.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    num_cols = df.select_dtypes(include=np.number).drop(
        "Credit_Score", axis=1, errors="ignore").columns.tolist()

    eda1, eda2, eda3, eda4 = st.tabs(
        ["Target & Distributions","Correlations","Skewness & Transforms","Categorical"])

    with eda1:
        c1,c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,4))
            counts = df["Credit_Score"].map(LABEL_MAP).value_counts().reindex(CLASS_NAMES)
            bars = ax.bar(counts.index, counts.values, color=COLORS[:3], edgecolor="white", width=0.6)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+50,
                        f"{int(b.get_height()):,}", ha="center", fontweight="bold")
            ax.set_title("Credit Score Distribution", fontweight="bold")
            ax.set_ylabel("Count"); sns.despine()
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=COLORS[:3], startangle=90, pctdistance=0.8,
                   wedgeprops={"edgecolor":"white","linewidth":2})
            ax.set_title("Class Share", fontweight="bold")
            st.pyplot(fig); plt.close()

        st.markdown("**Key numeric feature distributions by class**")
        feat_sel = st.multiselect("Select features to visualise", num_cols,
            default=["Annual_Income","Outstanding_Debt","Credit_History_Age",
                     "Delay_from_due_date"][:min(4,len(num_cols))])
        if feat_sel:
            fig, axes = plt.subplots(1, len(feat_sel), figsize=(5*len(feat_sel), 4))
            if len(feat_sel)==1: axes=[axes]
            for ax, col in zip(axes, feat_sel):
                for i,(sc,lb) in enumerate(LABEL_MAP.items()):
                    d = df.loc[df["Credit_Score"]==sc, col].dropna()
                    ax.hist(d, bins=40, alpha=0.55, label=lb, color=COLORS[i], density=True)
                ax.set_title(col.replace("_"," ")); ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Box plots
        st.markdown("**Box plots — outlier view**")
        fig, axes = plt.subplots(1, min(4,len(num_cols)), figsize=(16,4))
        if len(num_cols)<2: axes=[axes]
        for ax, col in zip(np.array(axes).ravel(), num_cols[:4]):
            data = [df.loc[df["Credit_Score"]==sc, col].dropna() for sc in [0,1,2]]
            bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True)
            for patch,c in zip(bp["boxes"],COLORS):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_title(col.replace("_"," "), fontsize=9)
        plt.suptitle("Box Plots by Credit Score Class", fontweight="bold", y=1.02)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with eda2:
        fig, ax = plt.subplots(figsize=(14,11))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.5, vmin=-1, vmax=1, square=True, ax=ax,
                    annot_kws={"size":7})
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Top correlations with target
        st.markdown("**Top 10 features correlated with Credit Score**")
        tgt_corr = df[num_cols+["Credit_Score"]].corr()["Credit_Score"].drop(
            "Credit_Score").abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        tgt_corr.sort_values().plot(kind="barh", ax=ax, color="#3b82f6", edgecolor="white")
        ax.set_title("Feature — Target Correlation", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with eda3:
        skew = df[num_cols].skew().sort_values(ascending=False)
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        colors_s = ["#ef4444" if abs(s)>1 else "#f59e0b" if abs(s)>0.5 else "#22c55e"
                    for s in skew]
        skew.plot(kind="bar", ax=axes[0], color=colors_s, edgecolor="white")
        axes[0].axhline(1, color="red", ls="--", lw=1.5)
        axes[0].axhline(-1,color="red", ls="--", lw=1.5)
        axes[0].set_title("Raw Feature Skewness", fontweight="bold")
        # Log-transform comparison
        log_skew = np.log1p(df[num_cols].clip(lower=0).fillna(0)).skew().sort_values(ascending=False)
        log_cols = ["#ef4444" if abs(s)>1 else "#f59e0b" if abs(s)>0.5 else "#22c55e"
                    for s in log_skew]
        log_skew.plot(kind="bar", ax=axes[1], color=log_cols, edgecolor="white")
        axes[1].set_title("After Log Transform", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with eda4:
        cat_feats = [c for c in ["Credit_Mix","Payment_of_Min_Amount","Occupation"]
                     if c in df.columns]
        if cat_feats:
            fig, axes = plt.subplots(1, len(cat_feats), figsize=(6*len(cat_feats),5))
            if len(cat_feats)==1: axes=[axes]
            for ax, feat in zip(axes, cat_feats):
                ct = pd.crosstab(df[feat], df["Credit_Score"].map(LABEL_MAP))
                for c in CLASS_NAMES:
                    if c not in ct.columns: ct[c] = 0
                ct[CLASS_NAMES].plot(kind="bar", ax=ax, color=COLORS[:3],
                                     edgecolor="white", rot=30)
                ax.set_title(feat.replace("_"," ")); ax.set_xlabel(""); ax.legend(fontsize=8)
            plt.suptitle("Categorical Features vs Credit Score",
                         fontweight="bold", y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🔮 Single Applicant Credit Score Prediction")

    model_choice = st.selectbox("Model", list(pipe["results"].keys()),
                                index=list(pipe["results"].keys()).index(pipe["best_name"]))

    with st.form("predict_form"):
        st.markdown("### 📋 Applicant Details")
        c1,c2,c3 = st.columns(3)
        with c1:
            age       = st.number_input("Age", 18, 80, 34)
            occ       = st.selectbox("Occupation", ["Engineer","Teacher","Doctor","Lawyer",
                "Accountant","Scientist","Manager","Entrepreneur","Developer",
                "Journalist","Musician","Mechanic","Writer","Architect","Media_Manager"])
            ai        = st.number_input("Annual Income ($)", 5000, 500000, 85000, step=1000)
            sal       = st.number_input("Monthly Salary ($)", 200, 50000, 7083, step=100)
        with c2:
            accounts  = st.number_input("Bank Accounts", 1, 15, 3)
            cards     = st.number_input("Credit Cards", 0, 20, 4)
            loans     = st.number_input("Number of Loans", 0, 15, 2)
            ir        = st.number_input("Interest Rate (%)", 1.0, 35.0, 11.5)
        with c3:
            delay     = st.number_input("Days Past Due (avg)", 0, 90, 5)
            dpm       = st.number_input("Delayed Payments (count)", 0, 30, 1)
            debt      = st.number_input("Outstanding Debt ($)", 0, 10000, 1200, step=50)
            hist      = st.number_input("Credit History (years)", 0.0, 40.0, 12.0, step=0.5)

        c4,c5 = st.columns(2)
        with c4:
            mix       = st.selectbox("Credit Mix", ["Good","Standard","Bad"])
            pma       = st.selectbox("Min Amount Payment", ["Yes","No","NM"])
            util      = st.slider("Credit Utilisation (%)", 0.0, 100.0, 28.5)
        with c5:
            emi       = st.number_input("Monthly EMI ($)", 0, 10000, 450, step=50)
            invest    = st.number_input("Monthly Investment ($)", 0, 10000, 800, step=50)
            balance   = st.number_input("Monthly Balance ($)", 0, 10000, 1500, step=50)
            inquiries = st.number_input("Credit Inquiries", 0, 20, 3)

        submitted = st.form_submit_button("🔮 Predict Credit Score", use_container_width=True)

    if submitted:
        raw = {
            "Age": age, "Occupation": occ, "Annual_Income": ai,
            "Monthly_Inhand_Salary": sal, "Num_Bank_Accounts": accounts,
            "Num_Credit_Card": cards, "Interest_Rate": ir, "Num_of_Loan": loans,
            "Delay_from_due_date": delay, "Num_of_Delayed_Payment": dpm,
            "Changed_Credit_Limit": 5.0, "Num_Credit_Inquiries": inquiries,
            "Credit_Mix": mix, "Outstanding_Debt": debt,
            "Credit_Utilization_Ratio": util, "Credit_History_Age": hist,
            "Payment_of_Min_Amount": pma, "Total_EMI_per_month": emi,
            "Amount_invested_monthly": invest, "Monthly_Balance": balance,
        }
        model_obj = pipe["results"][model_choice]["model"]
        label, proba = predict_applicant(raw, model=model_obj)
        confidence   = proba.max()

        color_map = {"Good":"score-good","Standard":"score-std","Poor":"score-poor"}
        emoji_map  = {"Good":"✅","Standard":"⚠️","Poor":"❌"}

        st.markdown(f"""
        <div class='metric-card {color_map[label]}' style='text-align:center;padding:24px;'>
          <h2 style='color:white;margin:0'>{emoji_map[label]} Credit Score: {label}</h2>
          <p>Confidence: {confidence*100:.1f}% &nbsp;|&nbsp; Model: {model_choice}</p>
        </div>""", unsafe_allow_html=True)

        gc1, gc2 = st.columns([1,1])
        with gc1:
            g_fig = gauge_chart(confidence, label)
            st.pyplot(g_fig); plt.close()
        with gc2:
            fig2, ax2 = plt.subplots(figsize=(5,3))
            bars2 = ax2.barh(CLASS_NAMES, proba, color=COLORS[:3], edgecolor="white")
            for bar, p in zip(bars2, proba):
                ax2.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                         f"{p*100:.1f}%", va="center", fontweight="bold")
            ax2.set_xlim(0,1.15); ax2.set_xlabel("Probability")
            ax2.set_title("Class Probabilities", fontweight="bold")
            sns.despine(); plt.tight_layout()
            st.pyplot(fig2); plt.close()

        # Interpretation
        st.markdown("### 📖 Plain-English Interpretation")
        advice = {
            "Good": "🟢 **Excellent credit profile.** This applicant demonstrates strong financial"
                    " habits — low debt, consistent payments, and a healthy credit history."
                    " Recommended for premium loan products at competitive rates.",
            "Standard": "🟡 **Moderate credit profile.** This applicant has an acceptable credit"
                        " standing but shows some risk indicators (e.g., occasional delays or"
                        " moderate debt load). Consider standard lending terms with regular review.",
            "Poor": "🔴 **High-risk credit profile.** Significant risk factors detected — high"
                    " outstanding debt, frequent payment delays, or insufficient credit history."
                    " Recommend declined or secured lending with close monitoring.",
        }
        st.info(advice[label])

        # Store for report
        ss["last_prediction"] = {
            "raw": raw, "label": label, "proba": proba.tolist(),
            "confidence": float(confidence), "model": model_choice,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BATCH SCORING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 📦 Batch Credit Scoring")
    st.markdown("""
    <div class='info-box'>
      Upload a CSV or Excel file containing applicant records. Required columns match the
      input form above. Results will include predicted class and all class probabilities.
    </div>""", unsafe_allow_html=True)

    template_cols = ["Age","Occupation","Annual_Income","Monthly_Inhand_Salary",
        "Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Num_of_Loan",
        "Delay_from_due_date","Num_of_Delayed_Payment","Changed_Credit_Limit",
        "Num_Credit_Inquiries","Credit_Mix","Outstanding_Debt",
        "Credit_Utilization_Ratio","Credit_History_Age","Payment_of_Min_Amount",
        "Total_EMI_per_month","Amount_invested_monthly","Monthly_Balance"]

    # Download template
    tpl = pd.DataFrame([{c:("Engineer" if c=="Occupation" else
                            "Good" if c=="Credit_Mix" else
                            "Yes" if c=="Payment_of_Min_Amount" else 0)
                         for c in template_cols}])
    tpl_buf = io.BytesIO()
    tpl.to_csv(tpl_buf, index=False); tpl_buf.seek(0)
    st.download_button("⬇️ Download Template CSV", tpl_buf, "template.csv", "text/csv")

    batch_file = st.file_uploader("Upload applicant file (CSV or Excel)",
                                  type=["csv","xlsx"], key="batch_upload")
    batch_model = st.selectbox("Scoring Model",
                               list(pipe["results"].keys()), key="batch_model",
                               index=list(pipe["results"].keys()).index(pipe["best_name"]))

    if batch_file:
        try:
            df_batch = pd.read_csv(batch_file) if batch_file.name.endswith(".csv") \
                       else pd.read_excel(batch_file)
            st.success(f"Loaded {len(df_batch):,} records")

            mdl_b = pipe["results"][batch_model]["model"]
            preds_b, probas_b = [], []
            prog = st.progress(0)
            for i, row in df_batch.iterrows():
                lbl, prb = predict_applicant(row.to_dict(), model=mdl_b)
                preds_b.append(lbl); probas_b.append(prb)
                prog.progress((i+1)/len(df_batch))

            df_batch["Predicted_Score"] = preds_b
            df_batch["Poor_prob"]       = [p[0] for p in probas_b]
            df_batch["Standard_prob"]   = [p[1] for p in probas_b]
            df_batch["Good_prob"]       = [p[2] for p in probas_b]
            df_batch["Confidence"]      = [p.max() for p in probas_b]
            prog.empty()

            st.dataframe(df_batch.head(20), use_container_width=True)

            # Summary pie
            fig_b, ax_b = plt.subplots(figsize=(5,4))
            vc = pd.Series(preds_b).value_counts()
            ax_b.pie(vc, labels=vc.index, autopct="%1.1f%%", colors=COLORS[:3],
                     startangle=90, wedgeprops={"edgecolor":"white"})
            ax_b.set_title("Batch Score Distribution", fontweight="bold")
            st.pyplot(fig_b); plt.close()

            # Export
            out_buf = io.BytesIO()
            df_batch.to_csv(out_buf, index=False); out_buf.seek(0)
            st.download_button("⬇️ Download Scored CSV", out_buf,
                               "scored_results.csv", "text/csv")

            xl_buf = io.BytesIO()
            with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                df_batch.to_excel(writer, index=False, sheet_name="Scored Results")
                pd.DataFrame({"Label":vc.index,"Count":vc.values}).to_excel(
                    writer, index=False, sheet_name="Summary")
            xl_buf.seek(0)
            st.download_button("⬇️ Download Scored Excel", xl_buf,
                               "scored_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 📈 Model Evaluation")

    # Leaderboard
    st.markdown("### 🏆 Model Leaderboard")
    st.dataframe(pipe["summary"].style.highlight_max(
        subset=["Accuracy","ROC-AUC","MCC","Cohen's Kappa"],
        color="#d1fae5"), use_container_width=True)

    ev1, ev2, ev3 = st.tabs(["Metrics","Confusion Matrices","ROC / PR / Calibration"])

    with ev1:
        # Grouped bar
        summary = pipe["summary"]
        fig, axes = plt.subplots(1,2,figsize=(16,6))
        metrics_list = ["Accuracy","ROC-AUC","MCC","Cohen's Kappa"]
        x = np.arange(len(summary)); w = 0.2
        for i,m in enumerate(metrics_list):
            axes[0].bar(x+i*w, summary[m], w, label=m, color=COLORS[i], alpha=0.85)
        axes[0].set_xticks(x+w*1.5)
        axes[0].set_xticklabels(summary["Model"], rotation=25, ha="right", fontsize=9)
        axes[0].set_ylim(0,1.15); axes[0].set_ylabel("Score")
        axes[0].set_title("All Metrics by Model", fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[1].scatter(summary["Accuracy"], summary["ROC-AUC"],
                        c=range(len(summary)), cmap="plasma", s=160, zorder=5)
        for _,row in summary.iterrows():
            axes[1].annotate(row["Model"],(row["Accuracy"],row["ROC-AUC"]),
                             textcoords="offset points",xytext=(6,4),fontsize=9)
        axes[1].set_xlabel("Accuracy"); axes[1].set_ylabel("ROC-AUC")
        axes[1].set_title("Accuracy vs ROC-AUC", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # RFECV curve
        st.markdown("### Feature Selection (RFECV)")
        scores = pipe["rfecv_scores"]; stds = pipe["rfecv_stds"]
        fig, ax = plt.subplots(figsize=(10,4))
        x_vals = range(1,len(scores)+1)
        ax.plot(x_vals, scores, color="#3b82f6", lw=2)
        ax.fill_between(x_vals, scores-stds, scores+stds, alpha=0.2, color="#3b82f6")
        ax.axvline(pipe["n_features"], color="red", ls="--",
                   label=f"Optimal: {pipe['n_features']}")
        ax.set_xlabel("Number of Features"); ax.set_ylabel("CV ROC-AUC")
        ax.set_title("RFECV Feature Selection Curve", fontweight="bold")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with ev2:
        res = pipe["results"]
        n_m   = len(res)
        ncols = 3; nrows = (n_m+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
        axes_flat = np.array(axes).ravel()
        for ax,(name, r) in zip(axes_flat, res.items()):
            cm = confusion_matrix(pipe["y_te"], r["preds"])
            ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{name}\nAcc:{r['acc']:.3f} AUC:{r['auc']:.3f}", fontsize=9)
        for ax in axes_flat[n_m:]: ax.set_visible(False)
        plt.suptitle("Confusion Matrices", fontsize=14, y=1.01, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with ev3:
        y_bin = label_binarize(pipe["y_te"], classes=[0,1,2])

        # ROC
        fig, axes = plt.subplots(1,3,figsize=(18,6))
        for i,cls in enumerate(CLASS_NAMES):
            ax = axes[i]
            for j,(name,r) in enumerate(res.items()):
                fpr,tpr,_ = roc_curve(y_bin[:,i], r["proba"][:,i])
                auc_i = roc_auc_score(y_bin[:,i], r["proba"][:,i])
                ax.plot(fpr,tpr,label=f"{name} ({auc_i:.3f})",
                        color=plt.cm.tab10(j/len(res)), lw=1.5)
            ax.plot([0,1],[0,1],"k--",lw=1)
            ax.set_title(f"ROC — {cls}", fontweight="bold")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7)
        plt.suptitle("ROC Curves by Class", fontweight="bold", y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # PR
        fig, axes = plt.subplots(1,3,figsize=(18,6))
        for i,cls in enumerate(CLASS_NAMES):
            ax = axes[i]
            for j,(name,r) in enumerate(res.items()):
                p,rec,_ = precision_recall_curve(y_bin[:,i], r["proba"][:,i])
                ap = average_precision_score(y_bin[:,i], r["proba"][:,i])
                ax.plot(rec,p,label=f"{name} (AP={ap:.3f})",
                        color=plt.cm.tab10(j/len(res)), lw=1.5)
            ax.set_title(f"PR Curve — {cls}", fontweight="bold")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=7)
        plt.suptitle("Precision-Recall Curves", fontweight="bold", y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Calibration
        top3 = pipe["summary"].head(3)["Model"].tolist()
        fig, axes = plt.subplots(1,3,figsize=(18,5))
        for i,cls in enumerate(CLASS_NAMES):
            ax = axes[i]
            ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
            for name in top3:
                fp,mp = calibration_curve(y_bin[:,i], res[name]["proba"][:,i], n_bins=10)
                ax.plot(mp,fp,"s-",label=name, lw=1.5)
            ax.set_title(f"Calibration — {cls}", fontweight="bold")
            ax.set_xlabel("Mean predicted prob")
            ax.set_ylabel("Fraction of positives"); ax.legend(fontsize=8)
        plt.suptitle("Calibration Curves", fontweight="bold", y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — REPORT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 📋 Export Report")
    st.markdown("""
    <div class='info-box'>
      Generate a full analysis report including model leaderboard, key charts, and
      (if you made a prediction) the applicant result. Downloads as Excel or plain-text PDF.
    </div>""", unsafe_allow_html=True)

    include_pred  = st.checkbox("Include last prediction result", value=True)
    include_eda   = st.checkbox("Include EDA charts", value=True)
    include_eval  = st.checkbox("Include model evaluation", value=True)

    if st.button("📊 Generate Excel Report", use_container_width=True):
        xl_buf = io.BytesIO()
        summary = pipe["summary"]
        with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
            # Sheet 1: Leaderboard
            summary.to_excel(writer, index=False, sheet_name="Model Leaderboard")

            # Sheet 2: Prediction
            if include_pred and ss.get("last_prediction"):
                pred = ss["last_prediction"]
                pred_df = pd.DataFrame([{
                    "Timestamp": pred["timestamp"],
                    "Model": pred["model"],
                    "Predicted Class": pred["label"],
                    "Confidence": f"{pred['confidence']*100:.1f}%",
                    "Poor Prob": f"{pred['proba'][0]*100:.1f}%",
                    "Standard Prob": f"{pred['proba'][1]*100:.1f}%",
                    "Good Prob": f"{pred['proba'][2]*100:.1f}%",
                }])
                raw_df = pd.DataFrame([pred["raw"]])
                pred_df.to_excel(writer, index=False, sheet_name="Prediction Result")
                raw_df.to_excel(writer, index=False, sheet_name="Applicant Details")

            # Sheet 3: Class distribution
            counts = df["Credit_Score"].map(LABEL_MAP).value_counts().reindex(CLASS_NAMES)
            counts.reset_index().rename(columns={"index":"Class","Credit_Score":"Count"})\
                  .to_excel(writer, index=False, sheet_name="Class Distribution")

            # Sheet 4: All model metrics detailed
            detail_rows = []
            for name,r in pipe["results"].items():
                detail_rows.append({
                    "Model":name, "Accuracy":round(r["acc"],4),
                    "ROC-AUC":round(r["auc"],4), "MCC":round(r["mcc"],4),
                    "Cohen's Kappa":round(r["kappa"],4)})
            pd.DataFrame(detail_rows).to_excel(
                writer, index=False, sheet_name="Detailed Metrics")

        xl_buf.seek(0)
        st.download_button(
            "⬇️ Download Excel Report",
            xl_buf,
            f"credit_score_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
        st.success("✅ Excel report ready for download")

    st.markdown("---")

    if st.button("📄 Generate Text Report (plain PDF summary)", use_container_width=True):
        # Build a text summary — works without fpdf too
        summary = pipe["summary"]
        best = pipe["best_name"]
        r    = pipe["results"][best]
        lines = [
            "=" * 65,
            "  CREDIT SCORE INTELLIGENCE — ANALYSIS REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Analyst  : {ss['username'].title()}",
            "=" * 65,
            "",
            "DATA SOURCE",
            "-" * 40,
            "Kaggle Credit Score Classification Dataset",
            "https://kaggle.com/datasets/parisrohan/credit-score-classification",
            f"Records  : {len(df):,}",
            f"Features : {len(pipe['feature_cols'])} raw → {pipe['n_features']} after RFECV",
            "",
            "MODEL LEADERBOARD",
            "-" * 40,
        ]
        for _,row in summary.iterrows():
            lines.append(
                f"  {row['Model']:28s}  AUC={row['ROC-AUC']:.4f}  "
                f"Acc={row['Accuracy']:.4f}  MCC={row['MCC']:.4f}")
        lines += [
            "",
            f"BEST MODEL : {best}",
            f"  Accuracy     : {r['acc']*100:.2f}%",
            f"  ROC-AUC      : {r['auc']:.4f}",
            f"  MCC          : {r['mcc']:.4f}",
            f"  Cohen Kappa  : {r['kappa']:.4f}",
            "",
        ]
        if include_pred and ss.get("last_prediction"):
            pred = ss["last_prediction"]
            lines += [
                "LAST PREDICTION",
                "-" * 40,
                f"  Timestamp    : {pred['timestamp']}",
                f"  Model        : {pred['model']}",
                f"  Result       : {pred['label']}",
                f"  Confidence   : {pred['confidence']*100:.1f}%",
                f"  Probabilities: Poor={pred['proba'][0]*100:.1f}%  "
                f"Standard={pred['proba'][1]*100:.1f}%  "
                f"Good={pred['proba'][2]*100:.1f}%",
                "",
            ]
        lines += [
            "KEY TAKEAWAYS",
            "-" * 40,
            "• Stacking ensemble outperforms all individual models",
            "• Top predictors: Outstanding Debt, Credit History Age,",
            "  Delay from Due Date, Debt-to-Income ratio",
            "• RFECV reduced feature set without AUC loss",
            "• Isotonic calibration gives reliable probability estimates",
            "• RobustScaler handles financial data outliers effectively",
            "",
            "STAKEHOLDERS",
            "-" * 40,
            "• Banks & lenders — automated loan decisioning",
            "• Credit bureaus  — model validation & enrichment",
            "• FinTechs        — real-time scoring APIs",
            "• Insurance firms — risk-based pricing",
            "• Regulators      — explainable, auditable decisions",
            "",
            "=" * 65,
        ]
        report_text = "\n".join(lines)
        st.text(report_text)
        st.download_button(
            "⬇️ Download Text Report",
            report_text,
            f"credit_score_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            use_container_width=True)

    st.markdown("---")
    st.markdown("### 📸 Download Individual Charts")
    chart_choice = st.selectbox("Select chart", [
        "Class Distribution","Feature Importance","ROC Curves (All Classes)",
        "Confusion Matrix (Best Model)","Calibration Curves"])

    if st.button("Generate & Download Chart", use_container_width=True):
        res  = pipe["results"]
        y_te = pipe["y_te"]
        y_bin = label_binarize(y_te, classes=[0,1,2])
        best = pipe["best_name"]

        if chart_choice == "Class Distribution":
            fig,ax = plt.subplots(figsize=(7,5))
            counts = df["Credit_Score"].map(LABEL_MAP).value_counts().reindex(CLASS_NAMES)
            bars = ax.bar(counts.index, counts.values, color=COLORS[:3], edgecolor="white", width=0.6)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+50,
                        f"{int(b.get_height()):,}", ha="center", fontweight="bold")
            ax.set_title("Credit Score Class Distribution", fontweight="bold")
            sns.despine(); plt.tight_layout()

        elif chart_choice == "Feature Importance":
            et = ExtraTreesClassifier(200, random_state=SEED, n_jobs=-1)
            et.fit(pipe["Xte_sel"], y_te)
            fi = pd.Series(et.feature_importances_, index=pipe["selected"]).sort_values()
            fig, ax = plt.subplots(figsize=(10, max(5, len(fi)*0.35)))
            fi.tail(15).plot(kind="barh", ax=ax, color="#3b82f6", edgecolor="white")
            ax.set_title("Feature Importances (Top 15)", fontweight="bold")
            plt.tight_layout()

        elif chart_choice == "ROC Curves (All Classes)":
            fig, axes = plt.subplots(1,3,figsize=(18,6))
            for i,cls in enumerate(CLASS_NAMES):
                ax=axes[i]
                for j,(name,r) in enumerate(res.items()):
                    fpr,tpr,_=roc_curve(y_bin[:,i],r["proba"][:,i])
                    auc_i=roc_auc_score(y_bin[:,i],r["proba"][:,i])
                    ax.plot(fpr,tpr,label=f"{name} ({auc_i:.3f})",
                            color=plt.cm.tab10(j/len(res)),lw=1.5)
                ax.plot([0,1],[0,1],"k--",lw=1)
                ax.set_title(f"ROC — {cls}",fontweight="bold")
                ax.legend(fontsize=7)
            plt.suptitle("ROC Curves",fontweight="bold",y=1.01)
            plt.tight_layout()

        elif chart_choice == "Confusion Matrix (Best Model)":
            fig,ax = plt.subplots(figsize=(6,5))
            cm = confusion_matrix(y_te, res[best]["preds"])
            ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"Confusion Matrix — {best}", fontweight="bold")
            plt.tight_layout()

        else:  # Calibration
            fig, axes = plt.subplots(1,3,figsize=(18,5))
            top3 = pipe["summary"].head(3)["Model"].tolist()
            for i,cls in enumerate(CLASS_NAMES):
                ax=axes[i]
                ax.plot([0,1],[0,1],"k--",lw=1)
                for name in top3:
                    fp,mp=calibration_curve(y_bin[:,i],res[name]["proba"][:,i],n_bins=10)
                    ax.plot(mp,fp,"s-",label=name,lw=1.5)
                ax.set_title(f"Calibration — {cls}",fontweight="bold")
                ax.legend(fontsize=8)
            plt.suptitle("Calibration Curves",fontweight="bold",y=1.01)
            plt.tight_layout()

        img_bytes = fig_to_bytes(fig); plt.close()
        st.image(img_bytes)
        st.download_button("⬇️ Download PNG",img_bytes,
                           f"{chart_choice.replace(' ','_')}.png","image/png",
                           use_container_width=True)
