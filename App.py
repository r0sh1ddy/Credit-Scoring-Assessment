
"""
CreditIQ - Production Streamlit App
Features: Auth · CSV Upload · Linearisation · Normalisation · Re-binning ·
          RFECV · SMOTE · Optuna · 9 Models · Voting · Stacking ·
          Gauge · Radar · Deviation · SHAP Waterfall · PDF Export
Run: streamlit run app.py
"""
 
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import io, hashlib, warnings, tempfile, os, math
from datetime import datetime
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
 
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import (LabelEncoder, StandardScaler, MinMaxScaler,
                                    RobustScaler, label_binarize)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score, roc_curve)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap
from fpdf import FPDF

import streamlit as st
from PIL import Image
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
app_icon = Image.open("credit score.png")

# Set it globally for the browser tab and app navigation
st.set_page_config(
    page_title="RiskEngine Pro | Credit Scoring ",
    page_icon=app_icon,
    layout="wide"
)

# Use it as a hero banner inside your app layout
st.image(app_icon, width=80)
st.title("Credit Risk Assessment Dashboard")
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
CLASS_NAMES = ["Poor", "Standard", "Good"]
LABEL_MAP   = {0: "Poor", 1: "Standard", 2: "Good"}
COLORS      = ["#ef4444", "#f59e0b", "#22c55e"]
LOG_FEATS   = ["Annual_Income","Monthly_Inhand_Salary","Outstanding_Debt",
               "Total_EMI_per_month","Amount_invested_monthly"]
CAT_COLS    = ["Occupation","Credit_Mix","Payment_of_Min_Amount",
               "Age_Group","Income_Bracket","Delay_Bucket"]
DARK_BG     = "#0f1117"
CARD_BG     = "#1a1f2e"
BORDER      = "#2a2f3e"
 
# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]         { font-family:'DM Sans',sans-serif; }
h1,h2,h3                         { font-family:'DM Serif Display',serif !important; }
.stApp                           { background:#0f1117; color:#e8e8e8; }
section[data-testid="stSidebar"] { background:#161b27; border-right:1px solid #2a2f3e; }
[data-testid="stMetric"]         { background:#1a1f2e; border:1px solid #2a2f3e;
                                   border-radius:12px; padding:16px 20px; }
[data-testid="stMetricValue"]    { font-size:1.7rem !important; font-weight:600; }
.score-badge { display:inline-block; padding:10px 32px; border-radius:50px;
               font-size:1.4rem; font-weight:700; font-family:'DM Serif Display',serif; }
.badge-good     { background:#052e16; color:#4ade80; border:2px solid #4ade80; }
.badge-standard { background:#1c1917; color:#fbbf24; border:2px solid #fbbf24; }
.badge-poor     { background:#1c0a0a; color:#f87171; border:2px solid #f87171; }
.info-box  { background:#1a1f2e; border-left:4px solid #3b82f6; padding:12px 16px;
             border-radius:0 8px 8px 0; margin:6px 0; font-size:.88rem; }
.warn-box  { background:#1c1917; border-left:4px solid #f59e0b; padding:12px 16px;
             border-radius:0 8px 8px 0; margin:6px 0; font-size:.88rem; }
.good-box  { background:#052e16; border-left:4px solid #22c55e; padding:12px 16px;
             border-radius:0 8px 8px 0; margin:6px 0; font-size:.88rem; }
.stButton>button { background:linear-gradient(135deg,#3b82f6,#1d4ed8); color:#fff;
                   border:none; border-radius:8px; font-weight:600; padding:10px 24px; }
hr { border-color:#2a2f3e !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
def _h(pw): return hashlib.sha256(pw.encode()).hexdigest()
 
USERS = {
    "admin":   {"hash": _h("admin123"),   "name": "Admin",          "role": "admin"},
    "analyst": {"hash": _h("analyst123"), "name": "Credit Analyst", "role": "analyst"},
    "demo":    {"hash": _h("demo"),       "name": "Demo User",      "role": "viewer"},
}
 
def login_page():
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 💳 CreditIQ")
        st.markdown("**Credit Score Analysis Platform**")
        st.markdown("---")
        username = st.text_input("Username", placeholder="admin / analyst / demo")
        password = st.text_input("Password", type="password")
        if st.button("Sign In", use_container_width=True):
            u = USERS.get(username)
            if u and u["hash"] == _h(password):
                for k, v in [("authenticated", True), ("username", username),
                              ("user_name", u["name"]), ("user_role", u["role"])]:
                    st.session_state[k] = v
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("""<div class="info-box"><b>Demo Credentials</b><br>
        👤 <code>admin</code> / <code>admin123</code><br>
        👤 <code>analyst</code> / <code>analyst123</code><br>
        👤 <code>demo</code> / <code>demo</code></div>""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dataset(n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    ai  = rng.lognormal(10.8, 0.6, n).round(2)
    od  = rng.uniform(0, 4998, n).round(2)
    dfd = rng.integers(0, 62, n).astype(float)
    ndp = rng.integers(0, 22, n).astype(float)
    cha = rng.uniform(0, 35, n).round(1)
    cm  = rng.choice(["Bad","Standard","Good"], n, p=[0.3, 0.4, 0.3])
    cu  = rng.uniform(20, 50, n).round(2)
    nba = rng.integers(1, 10, n).astype(float)
    ncc = rng.integers(0, 12, n).astype(float)
    for arr in [ai, od, dfd, ndp, nba]: arr[rng.random(n) < 0.05] = np.nan
    s = (0.25*(ai/np.nanmax(ai))
        +0.20*(1-np.where(np.isnan(od),0.5,od)/4998)
        +0.15*(1-np.where(np.isnan(dfd),0.5,dfd)/62)
        +0.12*(cha/35)
        +0.10*(1-np.where(np.isnan(ndp),0.5,ndp)/22)
        +0.08*np.where(cm=="Good",1.0,np.where(cm=="Standard",0.5,0.0))
        +0.05*(1-cu/50)+0.05*rng.uniform(0,1,n))
    labels = np.where(np.clip(s,0,1)<0.38,0,np.where(np.clip(s,0,1)<0.65,1,2))
    return pd.DataFrame({
        "Age": rng.integers(18,75,n).astype(float), "Occupation":
        rng.choice(["Scientist","Teacher","Engineer","Entrepreneur","Developer",
                    "Lawyer","Doctor","Journalist","Manager","Accountant",
                    "Musician","Mechanic","Writer","Architect","Media_Manager"],n),
        "Annual_Income": ai, "Monthly_Inhand_Salary": ai/12,
        "Num_Bank_Accounts": nba, "Num_Credit_Card": ncc,
        "Interest_Rate": rng.uniform(1,34,n).round(2),
        "Num_of_Loan": rng.integers(0,9,n).astype(float),
        "Delay_from_due_date": dfd, "Num_of_Delayed_Payment": ndp,
        "Changed_Credit_Limit": rng.uniform(0,30,n).round(2),
        "Num_Credit_Inquiries": rng.integers(0,17,n).astype(float),
        "Credit_Mix": cm, "Outstanding_Debt": od,
        "Credit_Utilization_Ratio": cu, "Credit_History_Age": cha,
        "Payment_of_Min_Amount": rng.choice(["Yes","No","NM"],n,p=[0.4,0.4,0.2]),
        "Total_EMI_per_month": rng.lognormal(4.5,0.8,n).round(2),
        "Amount_invested_monthly": rng.lognormal(5.0,1.0,n).round(2),
        "Monthly_Balance": rng.uniform(0,2000,n).round(2),
        "Credit_Score": labels,
    })
 
def load_upload(f):
    try:
        df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        if "Credit_Score" in df.columns and df["Credit_Score"].dtype == object:
            df["Credit_Score"] = df["Credit_Score"].map({"Poor":0,"Standard":1,"Good":2})
        return df
    except Exception as e:
        st.error(f"Upload error: {e}"); return None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def linearise(df):
    df = df.copy()
    for col in LOG_FEATS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df
 
def rebin(df):
    df = df.copy()
    df["Age_Group"] = pd.cut(df["Age"], bins=[17,25,35,45,60,75],
        labels=["18-25","26-35","36-45","46-60","61-75"]).astype(str)
    inc = np.expm1(df["Annual_Income"]) if df["Annual_Income"].max()<30 else df["Annual_Income"]
    df["Income_Bracket"] = pd.cut(inc, bins=[0,30000,70000,120000,1e9],
        labels=["Low","Medium","High","Very_High"]).astype(str)
    df["Delay_Bucket"] = pd.cut(df["Delay_from_due_date"].fillna(0),
        bins=[-1,0,15,30,62], labels=["None","Low","Medium","High"]).astype(str)
    return df
 
def engineer(df):
    df = df.copy()
    df["Debt_to_Income"]       = df["Outstanding_Debt"]       / (df["Annual_Income"]         + 1e-3)
    df["EMI_to_Income"]        = df["Total_EMI_per_month"]     / (df["Monthly_Inhand_Salary"] + 1e-3)
    df["Delayed_per_Loan"]     = df["Num_of_Delayed_Payment"]  / (df["Num_of_Loan"]           + 1)
    df["Investment_to_Income"] = df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + 1e-3)
    df["Cards_per_Account"]    = df["Num_Credit_Card"]         / (df["Num_Bank_Accounts"]     + 1)
    df["Util_x_Delay"]         = df["Credit_Utilization_Ratio"]* df["Delay_from_due_date"].fillna(0)
    df["Debt_x_Inquiries"]     = df["Outstanding_Debt"]        * df["Num_Credit_Inquiries"].fillna(0)
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  (cached per dataset hash)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_pipeline(_df: pd.DataFrame, df_key: str, n_trials: int = 25):
    df = linearise(_df)
    df = rebin(df)
 
    # Encode
    le_map = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_map[col] = le
 
    X_raw = df.drop("Credit_Score", axis=1)
    y     = df["Credit_Score"]
 
    # Impute
    imputer = SimpleImputer(strategy="median")
    X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
 
    # Engineer features
    X_fe = engineer(X_imp)
 
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fe, y, test_size=0.2, random_state=SEED, stratify=y)
 
    # Normalise
    scaler   = StandardScaler()
    X_tr_s   = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te_s   = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns)
 
    # RFECV
    rf_rfe = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=SEED, n_jobs=-1)
    rfecv  = RFECV(rf_rfe, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                   scoring="f1_macro", n_jobs=-1, min_features_to_select=8)
    rfecv.fit(X_tr_s, y_tr)
    sel    = X_tr_s.columns[rfecv.support_].tolist()
    X_tr_r = X_tr_s[sel]; X_te_r = X_te_s[sel]
 
    # SMOTE
    smote       = SMOTE(random_state=SEED)
    X_sm, y_sm  = smote.fit_resample(X_tr_r, y_tr)
 
    # Optuna XGBoost tuning
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",100,400),
                 max_depth=trial.suggest_int("max_depth",3,9),
                 learning_rate=trial.suggest_float("learning_rate",0.01,0.3,log=True),
                 subsample=trial.suggest_float("subsample",0.6,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
                 min_child_weight=trial.suggest_int("min_child_weight",1,10),
                 reg_alpha=trial.suggest_float("reg_alpha",1e-4,5.0,log=True),
                 reg_lambda=trial.suggest_float("reg_lambda",1e-4,5.0,log=True),
                 eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
        return cross_val_score(xgb.XGBClassifier(**p), X_sm, y_sm,
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
            scoring="roc_auc_ovr", n_jobs=-1).mean()
 
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best_p = {**study.best_params, "eval_metric":"mlogloss",
              "random_state":SEED, "n_jobs":-1}
 
    # All base models
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = {
        "Logistic Regression":  LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=4, random_state=SEED, n_jobs=-1),
        "Extra Trees":          ExtraTreesClassifier(n_estimators=200, max_depth=10,
                                    random_state=SEED, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                    learning_rate=0.08, random_state=SEED),
        "XGBoost (tuned)":      xgb.XGBClassifier(**best_p),
        "LightGBM":             lgb.LGBMClassifier(n_estimators=200, max_depth=6,
                                    learning_rate=0.05, random_state=SEED,
                                    n_jobs=-1, verbose=-1),
        "Naive Bayes":          GaussianNB(),
        "KNN":                  KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
        "SVM":                  SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
    }
 
    results = {}
    X_sm_np = X_sm if isinstance(X_sm, np.ndarray) else X_sm
    X_te_np = X_te_r.values
 
    for name, mdl in base.items():
        mdl.fit(X_sm_np, y_sm)
        p2 = mdl.predict(X_te_np); pr = mdl.predict_proba(X_te_np)
        cv = cross_val_score(mdl, X_sm_np, y_sm, cv=cv5,
                             scoring="roc_auc_ovr", n_jobs=-1)
        results[name] = dict(model=mdl, preds=p2, proba=pr,
            accuracy=accuracy_score(y_te,p2), f1=f1_score(y_te,p2,average="macro"),
            precision=precision_score(y_te,p2,average="macro"),
            recall=recall_score(y_te,p2,average="macro"),
            roc_auc=roc_auc_score(y_te,pr,multi_class="ovr",average="macro"),
            cv_mean=cv.mean(), cv_std=cv.std())
 
    # Voting ensemble
    top4 = sorted(results, key=lambda n: results[n]["roc_auc"], reverse=True)[:4]
    voting = VotingClassifier([(n, base[n]) for n in top4], voting="soft", n_jobs=-1)
    voting.fit(X_sm_np, y_sm)
    vp = voting.predict(X_te_np); vr = voting.predict_proba(X_te_np)
    cv_v = cross_val_score(voting, X_sm_np, y_sm, cv=cv5, scoring="roc_auc_ovr", n_jobs=-1)
    results["Voting Ensemble"] = dict(model=voting, preds=vp, proba=vr,
        accuracy=accuracy_score(y_te,vp), f1=f1_score(y_te,vp,average="macro"),
        precision=precision_score(y_te,vp,average="macro"),
        recall=recall_score(y_te,vp,average="macro"),
        roc_auc=roc_auc_score(y_te,vr,multi_class="ovr",average="macro"),
        cv_mean=cv_v.mean(), cv_std=cv_v.std())
 
    # Stacking
    stack = StackingClassifier([(n, base[n]) for n in top4],
        final_estimator=LogisticRegression(max_iter=2000, C=0.5, random_state=SEED),
        cv=5, n_jobs=-1)
    stack.fit(X_sm_np, y_sm)
    sp = stack.predict(X_te_np); sr = stack.predict_proba(X_te_np)
    cv_s = cross_val_score(stack, X_sm_np, y_sm, cv=cv5, scoring="roc_auc_ovr", n_jobs=-1)
    results["Stacking"] = dict(model=stack, preds=sp, proba=sr,
        accuracy=accuracy_score(y_te,sp), f1=f1_score(y_te,sp,average="macro"),
        precision=precision_score(y_te,sp,average="macro"),
        recall=recall_score(y_te,sp,average="macro"),
        roc_auc=roc_auc_score(y_te,sr,multi_class="ovr",average="macro"),
        cv_mean=cv_s.mean(), cv_std=cv_s.std())
 
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
 
    return dict(
        results=results, best_name=best_name, selected=sel, rfecv=rfecv,
        study=study, best_p=best_p, scaler=scaler, imputer=imputer,
        le_map=le_map, base=base, top4=top4,
        X_tr_r=X_tr_r, X_te_r=X_te_r, X_sm=X_sm,
        y_tr=y_tr, y_te=y_te, y_sm=y_sm,
        good_profile=X_tr_r[y_tr.values==2].mean().to_dict(),
        poor_profile=X_tr_r[y_tr.values==0].mean().to_dict(),
        smote_counts=dict(zip(*np.unique(y_sm, return_counts=True))),
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREDICT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def proba_to_score(proba):
    return int(300 + proba[1]*200 + proba[2]*550)
 
def expected_range(annual_income):
    if annual_income < 30000:  return 400, 560
    if annual_income < 80000:  return 540, 700
    if annual_income < 150000: return 650, 780
    return 700, 820
 
def radar_dims(inp):
    return {
        "Payment\nHistory":     max(0, 100-(inp.get("Num_of_Delayed_Payment",0) or 0)/22*80
                                       -(inp.get("Delay_from_due_date",0) or 0)/62*20),
        "Debt\nManagement":     max(0, 100-(inp.get("Outstanding_Debt",0) or 0)
                                         /(max(inp.get("Annual_Income",1),1))*100),
        "Credit\nHistory":      min(100,(inp.get("Credit_History_Age",0) or 0)/35*100),
        "Credit\nMix":          {"Good":100,"Standard":55,"Bad":10}.get(
                                 inp.get("Credit_Mix","Bad"),10),
        "Credit\nActivity":     max(0,100
                                   -(inp.get("Num_Credit_Inquiries",0) or 0)/17*50
                                   -max(0,(inp.get("Credit_Utilization_Ratio",30) or 30)-30)/20*50),
        "Savings &\nIncome":    min(100,(inp.get("Amount_invested_monthly",0) or 0)
                                       /max((inp.get("Monthly_Inhand_Salary",1) or 1),1)*80
                                       +min(20,(inp.get("Annual_Income",0) or 0)/150000*20)),
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def gauge(score, lo, hi, label):
    col = "#4ade80" if score>=740 else "#fbbf24" if score>=580 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference":(lo+hi)//2, "valueformat":".0f",
               "increasing":{"color":"#4ade80"}, "decreasing":{"color":"#ef4444"}},
        title={"text":f"<b>{label}</b><br><span style='font-size:.8em;color:#888'>"
                      f"Expected {lo}–{hi}</span>", "font":{"color":"#e8e8e8"}},
        number={"font":{"color":col,"size":50}},
        gauge={"axis":{"range":[300,850],"tickcolor":"#888","tickfont":{"color":"#888"}},
               "bar":{"color":col,"thickness":0.28},
               "bgcolor":CARD_BG, "bordercolor":BORDER,
               "steps":[{"range":[300,580],"color":"#2d0a0a"},
                         {"range":[580,670],"color":"#2a1f08"},
                         {"range":[670,740],"color":"#1a2008"},
                         {"range":[740,800],"color":"#052e16"},
                         {"range":[800,850],"color":"#013320"}],
               "threshold":{"line":{"color":"#60a5fa","width":5},
                             "thickness":0.85,"value":(lo+hi)//2}}))
    fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font={"color":"#e8e8e8"}, margin=dict(l=20,r=20,t=80,b=20), height=310)
    return fig
 
def radar(scores):
    cats = list(scores.keys()); vals = list(scores.values())
    vals += [vals[0]]; cats += [cats[0]]
    fig = go.Figure()
    fig.add_scatterpolar(r=[70]*len(cats), theta=cats, fill="toself",
        fillcolor="rgba(34,197,94,0.06)", line=dict(color="#22c55e",width=1,dash="dot"),
        name="Good target")
    fig.add_scatterpolar(r=vals, theta=cats, fill="toself",
        fillcolor="rgba(59,130,246,0.2)", line=dict(color="#3b82f6",width=2),
        name="Your profile")
    fig.update_layout(polar=dict(
        radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="#888"),gridcolor=BORDER),
        angularaxis=dict(tickfont=dict(color="#e8e8e8"),gridcolor=BORDER), bgcolor=CARD_BG),
        paper_bgcolor=CARD_BG, legend=dict(font=dict(color="#e8e8e8"),bgcolor=CARD_BG),
        margin=dict(l=60,r=60,t=40,b=40), height=330)
    return fig
 
def deviation_chart(user_fe, good_p, poor_p):
    KEYS = [("Outstanding_Debt",True,"Outstanding Debt"),
            ("Delay_from_due_date",True,"Avg Delay (days)"),
            ("Num_of_Delayed_Payment",True,"Delayed Payments"),
            ("Credit_History_Age",False,"Credit History (yrs)"),
            ("Credit_Utilization_Ratio",True,"Credit Utilisation %"),
            ("Debt_to_Income",True,"Debt-to-Income"),
            ("EMI_to_Income",True,"EMI-to-Income"),
            ("Annual_Income",False,"Annual Income")]
    rows = []
    for col, lower_better, lbl in KEYS:
        u = user_fe.get(col); g = good_p.get(col); p = poor_p.get(col)
        if u is None or g is None: continue
        rng = abs(g - p) if p else abs(g)+1
        pct = (u - g)/(rng+1e-9)*100
        if lower_better: pct = -pct
        rows.append({"Feature":lbl,"Deviation":pct,"User":round(u,2),"Good":round(g,2)})
    df_b = pd.DataFrame(rows).sort_values("Deviation")
    fig = go.Figure(go.Bar(
        x=df_b["Deviation"], y=df_b["Feature"], orientation="h",
        marker_color=["#ef4444" if d<0 else "#22c55e" for d in df_b["Deviation"]],
        hovertext=[f"{r['Feature']}<br>Yours: {r['User']}<br>Good avg: {r['Good']}"
                   for _,r in df_b.iterrows()], hoverinfo="text"))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.update_layout(title="Deviation from 'Good' Profile",
        xaxis_title="Relative deviation (%)", paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG, font=dict(color="#e8e8e8"),
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
        height=360, margin=dict(l=10,r=10,t=50,b=30))
    return fig
 
def shap_waterfall(mdl, x_single, feat_names, cls_idx):
    try:
        ex  = shap.TreeExplainer(mdl)
        raw = ex.shap_values(x_single)
        sv  = raw[0,:,cls_idx] if (isinstance(raw,np.ndarray) and raw.ndim==3) else raw[cls_idx][0]
        s   = pd.Series(sv, index=feat_names).abs().nlargest(12).index
        top = pd.Series(sv, index=feat_names)[s].sort_values()
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
            marker_color=["#ef4444" if v<0 else "#22c55e" for v in top.values]))
        fig.add_vline(x=0, line_color="#888", line_width=1)
        fig.update_layout(title=f"SHAP - Why '{CLASS_NAMES[cls_idx]}'?",
            xaxis_title="SHAP value", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color="#e8e8e8"), xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER), height=360, margin=dict(l=10,r=10,t=50,b=30))
        return fig
    except Exception as e:
        return None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _gauge_png(score, lo, hi):
    fig, ax = plt.subplots(figsize=(5,3), subplot_kw={"projection":"polar"}, facecolor="#1a1f2e")
    ax.set_facecolor("#1a1f2e")
    zones = [(300,580,"#4a1515"),(580,670,"#4a3d0a"),(670,740,"#2a3d0a"),(740,800,"#0a3d1a"),(800,850,"#0a4d22")]
    for zl,zh,zc in zones:
        t1 = math.pi*(1-(zl-300)/550); t2 = math.pi*(1-(zh-300)/550)
        ax.fill_between(np.linspace(t1,t2,40), 0.6, 1.0, color=zc, alpha=0.85)
    ang = math.pi*(1-(score-300)/550)
    ax.annotate("", xy=(ang,0.85), xytext=(ang,0.05),
                arrowprops=dict(arrowstyle="-|>",color="white",lw=2.5))
    ax.set_thetamin(0); ax.set_thetamax(180); ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(0,1)
    col = "#4ade80" if score>=740 else "#fbbf24" if score>=580 else "#ef4444"
    ax.text(math.pi/2, 0.32, str(score), ha="center", va="center",
            color=col, fontsize=20, fontweight="bold")
    ax.text(math.pi/2, 0.12, f"Expected: {lo}–{hi}", ha="center", color="#aaa", fontsize=7)
    buf = io.BytesIO()
    plt.tight_layout(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                                    facecolor="#1a1f2e"); plt.close(); buf.seek(0); return buf
 
def _radar_png(scores):
    cats = [c.replace("\n"," ") for c in scores]; vals = list(scores.values())
    N = len(cats); angles = [n/N*2*math.pi for n in range(N)]
    angles += angles[:1]; vals += vals[:1]
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={"polar":True}, facecolor="#1a1f2e")
    ax.set_facecolor("#1a1f2e"); ax.plot(angles,vals,color="#3b82f6",lw=2)
    ax.fill(angles,vals,color="#3b82f6",alpha=0.2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,color="#e8e8e8",fontsize=7)
    ax.set_ylim(0,100); ax.set_yticks([25,50,75]); ax.yaxis.set_tick_params(labelsize=6,labelcolor="#888")
    ax.grid(color="#2a2f3e"); ax.spines["polar"].set_color("#2a2f3e")
    buf = io.BytesIO()
    plt.tight_layout(); fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",
                                    facecolor="#1a1f2e"); plt.close(); buf.seek(0); return buf
 
def build_pdf(inputs, score, lo, hi, label, proba, tips, radar_sc, user_fe, good_p, poor_p):
    pdf = FPDF(); pdf.add_page(); pdf.set_margins(15,15,15)
    # header
    pdf.set_fill_color(15,17,23); pdf.rect(0,0,210,38,"F")
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",20)
    pdf.cell(0,16,"CreditIQ - Credit Score Report",ln=True,align="C")
    pdf.set_font("Helvetica","",9); pdf.set_text_color(160,160,160)
    pdf.cell(0,7,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
               f"Model: Stacking / XGBoost Tuned",ln=True,align="C")
    pdf.ln(6)
    # score banner
    rc = {"Poor":(200,50,50),"Standard":(220,150,30),"Good":(30,180,70)}[label]
    pdf.set_fill_color(*rc); pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",15)
    pdf.cell(0,11,f"  Credit Score: {label}   ({score} / 850)   "
               f"Expected Range: {lo}–{hi}",ln=True,fill=True,align="C"); pdf.ln(3)
    # gauge + radar
    gbuf = _gauge_png(score,lo,hi); rbuf = _radar_png(radar_sc)
    with tempfile.NamedTemporaryFile(suffix=".png",delete=False) as gf: gf.write(gbuf.read()); gp=gf.name
    with tempfile.NamedTemporaryFile(suffix=".png",delete=False) as rf: rf.write(rbuf.read()); rp=rf.name
    y0 = pdf.get_y()
    pdf.image(gp, x=15, y=y0, w=88); pdf.image(rp, x=108, y=y0, w=88)
    pdf.set_y(y0+58); os.unlink(gp); os.unlink(rp)
    # probabilities
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,30,30)
    pdf.cell(0,7,"Class Probabilities",ln=True); pdf.set_font("Helvetica","",9)
    for i,(cn,p) in enumerate(zip(CLASS_NAMES,proba)):
        rc2 = [(200,50,50),(220,150,30),(30,180,70)][i]
        pdf.set_fill_color(*rc2); pdf.cell(40,6,cn)
        pdf.cell(int(p*100),5,"",fill=True); pdf.cell(0,5,f"  {p:.1%}",ln=True)
    pdf.ln(3)
    # inputs table
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,30,30)
    pdf.cell(0,7,"Input Parameters",ln=True); pdf.set_font("Helvetica","",8)
    items = list(inputs.items())
    for j in range(0,len(items),2):
        k1,v1 = items[j]; k2,v2 = items[j+1] if j+1<len(items) else ("","")
        pdf.set_fill_color(240,240,240); pdf.set_text_color(80,80,80)
        pdf.cell(45,5,k1.replace("_"," "),fill=True)
        pdf.set_text_color(20,20,20); pdf.cell(45,5,str(v1))
        if k2:
            pdf.set_fill_color(240,240,240); pdf.set_text_color(80,80,80)
            pdf.cell(45,5,k2.replace("_"," "),fill=True)
            pdf.set_text_color(20,20,20); pdf.cell(0,5,str(v2),ln=True)
        else: pdf.ln()
    pdf.ln(3)
    # tips
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,30,30)
    pdf.cell(0,7,"Personalised Insights",ln=True); pdf.set_font("Helvetica","",8)
    for tip in tips:
        clean = tip.replace("⚠️","[!]").replace("✅","[OK]").replace("💡","[>]")
        pdf.multi_cell(0,5,clean)
    # footer
    pdf.set_y(-16); pdf.set_font("Helvetica","I",7); pdf.set_text_color(160,160,160)
    pdf.cell(0,5,"CreditIQ - For analytical purposes only. Not financial advice.",align="C")
    out = io.BytesIO(); pdf.output(out); out.seek(0); return out
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 💳 CreditIQ")
        st.caption(f"**{st.session_state.get('user_name','-')}** "
                   f"({st.session_state.get('user_role','-')})")
        st.markdown("---")
        page = st.radio("Navigate",
            ["🏠 Overview","📊 EDA","🤖 Models","🔮 Predict Score"],
            label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**📂 Upload Dataset**")
        up = st.file_uploader("CSV / Excel", type=["csv","xlsx"],
                               label_visibility="collapsed")
        if up:
            df_up = load_upload(up)
            if df_up is not None:
                st.session_state["df_uploaded"] = df_up
                st.success(f"Loaded {len(df_up):,} rows")
        if st.button("↩ Reset to Synthetic"):
            st.session_state.pop("df_uploaded", None)
            st.cache_resource.clear()
        st.markdown("---")
        n_trials = st.slider("Optuna trials", 10, 60, 25, 5)
        st.session_state["n_trials"] = n_trials
        st.markdown("---")
        if st.button("🚪 Sign Out"):
            for k in ["authenticated","username","user_name","user_role","df_uploaded"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.caption("XGBoost · LightGBM · SMOTE · RFECV · Optuna · SHAP")
    return page
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(df, pipe):
    st.title("Credit Score Analysis Dashboard")
    res = pipe["results"]; bn = pipe["best_name"]
    best = res[bn]
 
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{len(pipe['X_tr_r'].columns)} selected / {len(pipe['X_tr_r'].columns)+len(pipe['selected'])} total")
    c3.metric("Best Accuracy", f"{best['accuracy']:.1%}")
    c4.metric("Best ROC-AUC",  f"{best['roc_auc']:.4f}")
    c5.metric("Models Trained", str(len(res)))
    st.markdown("---")
 
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("Class Distribution")
        counts = df["Credit_Score"].map(LABEL_MAP).value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
            hole=0.5, marker_colors=COLORS, hoverinfo="label+percent+value"))
        fig.update_layout(paper_bgcolor=DARK_BG, font=dict(color="#e8e8e8"),
            height=270, margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(font=dict(color="#e8e8e8")))
        st.plotly_chart(fig, use_container_width=True)
 
    with col2:
        st.subheader("SMOTE Rebalancing")
        sc = pipe["smote_counts"]
        raw = dict(df["Credit_Score"].value_counts().sort_index())
        rows = ([{"Class":f"{LABEL_MAP[k]} before","Count":raw.get(k,0),"Phase":"Before"}
                 for k in sorted(raw)] +
                [{"Class":f"{LABEL_MAP.get(int(k),k)} after","Count":v,"Phase":"After SMOTE"}
                 for k,v in sc.items()])
        fig2 = px.bar(pd.DataFrame(rows), x="Class", y="Count", color="Phase",
            color_discrete_sequence=["#4C72B0","#22c55e"], barmode="group")
        fig2.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color="#e8e8e8"), height=270,
            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
            margin=dict(l=10,r=10,t=10,b=10),legend=dict(font=dict(color="#e8e8e8")))
        st.plotly_chart(fig2, use_container_width=True)
 
    with col3:
        st.subheader("Leaderboard")
        lb = pd.DataFrame({n: {"Acc":f'{v["accuracy"]:.4f}',"AUC":f'{v["roc_auc"]:.4f}',
             "F1":f'{v["f1"]:.4f}',"CV":f'{v["cv_mean"]:.3f}±{v["cv_std"]:.3f}'}
             for n,v in res.items()}).T.sort_values("AUC",ascending=False)
        st.dataframe(lb, use_container_width=True, height=270)
 
    st.markdown("---")
    st.subheader("Pipeline Summary")
    st.markdown(f"""
    <div class="info-box">
    📐 <b>Preprocessing:</b> Log-transform ({", ".join(LOG_FEATS[:3])}…) - Age/Income/Delay re-binning - Label encoding - Median imputation - StandardScaler<br>
    🔬 <b>Feature selection:</b> Mutual info - RF importance - RFECV - <b>{len(pipe["selected"])} features selected</b><br>
    ⚖️ <b>SMOTE:</b> Balanced training set - {sum(pipe["smote_counts"].values()):,} samples<br>
    🔧 <b>Optuna:</b> {st.session_state.get("n_trials",25)} trials · best CV AUC {pipe["study"].best_value:.4f}<br>
    🤖 <b>Models:</b> 9 base classifiers + Soft Voting + Stacking (meta: LR) · Best: <b>{bn}</b>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True, height=240)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
def page_eda(df):
    st.title("Exploratory Data Analysis")
    t1,t2,t3,t4,t5 = st.tabs(
        ["Distributions","Skewness & Linearisation","Categorical","Correlation","Missing Values"])
 
    with t1:
        num_cols = df.select_dtypes(include=np.number).columns.drop("Credit_Score").tolist()
        sel = st.multiselect("Features", num_cols, default=num_cols[:6])
        if sel:
            nc=3; nr=math.ceil(len(sel)/nc)
            fig,axes=plt.subplots(nr,nc,figsize=(16,4*nr),facecolor=DARK_BG)
            axes=np.array(axes).ravel()
            for ax,feat in zip(axes,sel):
                ax.set_facecolor(CARD_BG)
                for i,(s,l) in enumerate(LABEL_MAP.items()):
                    ax.hist(df.loc[df["Credit_Score"]==s,feat].dropna(),
                            bins=40,alpha=0.6,label=l,color=COLORS[i],density=True)
                ax.set_title(feat.replace("_"," "),color="#e8e8e8")
                ax.tick_params(colors="#888"); ax.legend(fontsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            for ax in axes[len(sel):]: ax.set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()
 
    with t2:
        st.subheader("Skewness - Before vs After Log Transform")
        skew_df = df[num_cols].skew().sort_values(ascending=False)
        log_df  = df[num_cols].copy()
        for c in LOG_FEATS:
            if c in log_df.columns: log_df[c] = np.log1p(log_df[c].clip(lower=0))
        skew_after = log_df.skew().sort_values(ascending=False)
        comp = pd.DataFrame({"Before":skew_df,"After Log":skew_after}).fillna(0)
        fig = go.Figure()
        fig.add_bar(name="Before", x=comp.index, y=comp["Before"], marker_color="#ef4444")
        fig.add_bar(name="After Log", x=comp.index, y=comp["After Log"], marker_color="#22c55e")
        fig.add_hline(y=1, line_dash="dot", line_color="orange"); fig.add_hline(y=-1, line_dash="dot", line_color="orange")
        fig.update_layout(barmode="group", title="Skewness Comparison (|>1| = high skew)",
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG, font=dict(color="#e8e8e8"),
            xaxis=dict(gridcolor=BORDER,tickangle=35), yaxis=dict(gridcolor=BORDER), height=380)
        st.plotly_chart(fig, use_container_width=True)
 
    with t3:
        cat_cols = df.select_dtypes(include=object).columns.tolist()
        if cat_cols:
            feat = st.selectbox("Feature", cat_cols)
            ct = pd.crosstab(df[feat], df["Credit_Score"].map(LABEL_MAP))
            for c in CLASS_NAMES:
                if c not in ct.columns: ct[c]=0
            fig2 = go.Figure()
            for i,cls in enumerate(CLASS_NAMES):
                fig2.add_bar(x=ct.index, y=ct[cls], name=cls, marker_color=COLORS[i])
            fig2.update_layout(barmode="group", paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                font=dict(color="#e8e8e8"), height=380,
                xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig2, use_container_width=True)
 
    with t4:
        corr = df.select_dtypes(include=np.number).drop("Credit_Score",axis=1).corr()
        fig3,ax3 = plt.subplots(figsize=(14,11),facecolor=DARK_BG)
        ax3.set_facecolor(CARD_BG)
        sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),annot=True,fmt=".2f",
            cmap="RdYlGn",linewidths=0.5,vmin=-1,vmax=1,square=True,ax=ax3,
            annot_kws={"size":7,"color":"#e8e8e8"})
        ax3.tick_params(colors="#888")
        plt.tight_layout(); st.pyplot(fig3); plt.close()
 
    with t5:
        miss=(df.isnull().mean()*100).sort_values(ascending=False)
        miss=miss[miss>0]
        if miss.empty:
            st.info("No missing values in this dataset.")
        else:
            fig4=go.Figure(go.Bar(x=miss.values,y=miss.index,orientation="h",marker_color="#DD8452"))
            fig4.update_layout(xaxis_title="Missing (%)",paper_bgcolor=DARK_BG,plot_bgcolor=CARD_BG,
                font=dict(color="#e8e8e8"),height=380,
                xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig4,use_container_width=True)
            st.dataframe(miss.rename("Missing %").round(2).to_frame(),use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODELS
# ─────────────────────────────────────────────────────────────────────────────
def page_models(pipe):
    st.title("Model Training & Evaluation")
    res=pipe["results"]; bn=pipe["best_name"]
    t1,t2,t3,t4,t5,t6 = st.tabs(
        ["Leaderboard","Confusion Matrices","ROC Curves",
         "Calibration","Optuna Study","RFECV & SHAP"])
 
    with t1:
        st.subheader("All Models - Full Metrics")
        lb = pd.DataFrame({n:{"Accuracy":round(v["accuracy"],4),"F1 Macro":round(v["f1"],4),
             "Precision":round(v["precision"],4),"Recall":round(v["recall"],4),
             "ROC-AUC":round(v["roc_auc"],4),
             "CV AUC":f'{v["cv_mean"]:.4f}±{v["cv_std"]:.3f}'}
             for n,v in res.items()}).T.sort_values("ROC-AUC",ascending=False)
        st.dataframe(lb, use_container_width=True)
        st.markdown(f"**🏆 Best model:** `{bn}`  |  AUC = `{res[bn]['roc_auc']:.4f}`")
 
        fig=go.Figure()
        names=list(lb.index)
        for metric,color in [("Accuracy","#4C72B0"),("F1 Macro","#55A868"),
                              ("Precision","#DD8452"),("ROC-AUC","#C44E52")]:
            fig.add_bar(name=metric, x=names, y=lb[metric].astype(float),
                        marker_color=color, opacity=0.85)
        fig.update_layout(barmode="group", paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color="#e8e8e8"), height=400, xaxis=dict(gridcolor=BORDER,tickangle=30),
            yaxis=dict(gridcolor=BORDER,range=[0,1.1]), legend=dict(font=dict(color="#e8e8e8")))
        st.plotly_chart(fig, use_container_width=True)
 
    with t2:
        mc = st.selectbox("Model", list(res.keys()), index=list(res.keys()).index(bn))
        cm = confusion_matrix(pipe["y_te"], res[mc]["preds"])
        fig2,ax2 = plt.subplots(figsize=(6,5),facecolor=DARK_BG)
        ax2.set_facecolor(CARD_BG)
        ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(ax=ax2,colorbar=False,cmap="Blues")
        ax2.set_title(f"{mc} - Acc: {res[mc]['accuracy']:.3f}",color="#e8e8e8")
        ax2.tick_params(colors="#888")
        for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.code(classification_report(pipe["y_te"],res[mc]["preds"],target_names=CLASS_NAMES))
 
    with t3:
        st.subheader(f"ROC Curves - {bn}")
        ybin = label_binarize(pipe["y_te"], classes=[0,1,2])
        fig3 = go.Figure()
        top5 = list(pd.DataFrame({n:{"auc":v["roc_auc"]} for n,v in res.items()}).T.sort_values("auc",ascending=False).head(5).index)
        cmap = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
        for i,nm in enumerate(top5):
            for ci,cn in enumerate(CLASS_NAMES):
                fpr,tpr,_=roc_curve(ybin[:,ci],res[nm]["proba"][:,ci])
                auc_i=roc_auc_score(ybin[:,ci],res[nm]["proba"][:,ci])
                fig3.add_scatter(x=fpr,y=tpr,line=dict(color=cmap[i],width=1.5),
                    opacity=0.7,name=f"{nm[:10]}/{cn} ({auc_i:.3f})")
        fig3.add_scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="#555"))
        fig3.update_layout(xaxis_title="FPR",yaxis_title="TPR",paper_bgcolor=DARK_BG,
            plot_bgcolor=CARD_BG,font=dict(color="#e8e8e8"),height=450,
            xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER),
            legend=dict(font=dict(color="#e8e8e8",size=8),bgcolor=DARK_BG))
        st.plotly_chart(fig3, use_container_width=True)
 
    with t4:
        st.subheader(f"Calibration Curves - {bn}")
        ybin2 = label_binarize(pipe["y_te"],classes=[0,1,2])
        fig4,axes4 = plt.subplots(1,3,figsize=(15,5),facecolor=DARK_BG)
        for i,(ax,cn) in enumerate(zip(axes4,CLASS_NAMES)):
            ax.set_facecolor(CARD_BG)
            fp,mp = calibration_curve(ybin2[:,i],res[bn]["proba"][:,i],n_bins=10)
            ax.plot(mp,fp,"s-",color="#4C72B0",label=bn[:15])
            ax.plot([0,1],[0,1],"k--",label="Perfect"); ax.legend(fontsize=7)
            ax.set_title(f"Class: {cn}",color="#e8e8e8"); ax.tick_params(colors="#888")
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.suptitle(f"Calibration - {bn}",color="#e8e8e8")
        plt.tight_layout(); st.pyplot(fig4); plt.close()
 
    with t5:
        st.subheader("Optuna Hyperparameter Search")
        study = pipe["study"]
        st.metric("Best CV ROC-AUC", f"{study.best_value:.4f}")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Best XGBoost Parameters:**")
            st.json(pipe["best_p"])
        with c2:
            tdf = study.trials_dataframe()
            fig5=px.line(tdf,x="number",y="value",title="Optuna Trial History",
                markers=True)
            fig5.update_layout(paper_bgcolor=DARK_BG,plot_bgcolor=CARD_BG,
                font=dict(color="#e8e8e8"),height=300,
                xaxis=dict(gridcolor=BORDER,title="Trial"),
                yaxis=dict(gridcolor=BORDER,title="CV ROC-AUC"))
            st.plotly_chart(fig5,use_container_width=True)
        try:
            imp=optuna.importance.get_param_importances(study)
            imp_df=pd.DataFrame({"Parameter":list(imp.keys()),"Importance":list(imp.values())})
            fig6=px.bar(imp_df.sort_values("Importance"),x="Importance",y="Parameter",
                orientation="h",title="Hyperparameter Importance")
            fig6.update_layout(paper_bgcolor=DARK_BG,plot_bgcolor=CARD_BG,
                font=dict(color="#e8e8e8"),height=300,
                xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig6,use_container_width=True)
        except Exception: pass
 
    with t6:
        c1,c2=st.columns(2)
        with c1:
            st.subheader("RFECV Feature Selection")
            st.markdown(f"**Selected {len(pipe['selected'])} optimal features:**")
            for f in sorted(pipe["selected"]): st.markdown(f"- `{f}`")
            cv_r=pipe["rfecv"].cv_results_
            ns=range(pipe["rfecv"].min_features_to_select,
                     pipe["rfecv"].min_features_to_select+len(cv_r["mean_test_score"]))
            fig7=go.Figure()
            fig7.add_scatter(x=list(ns),y=cv_r["mean_test_score"],mode="lines+markers",
                line=dict(color="#4C72B0",width=2))
            fig7.add_vline(x=len(pipe["selected"]),line_color="#ef4444",line_dash="dash")
            fig7.update_layout(title="RFECV CV Score vs Feature Count",
                xaxis_title="# Features",yaxis_title="CV F1 Macro",
                paper_bgcolor=DARK_BG,plot_bgcolor=CARD_BG,
                font=dict(color="#e8e8e8"),height=300,
                xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig7,use_container_width=True)
 
        with c2:
            st.subheader("SHAP Summary")
            xgb_mdl = pipe["base"].get("XGBoost (tuned)", pipe["base"].get("LightGBM"))
            N=300; Xs=pipe["X_te_r"].values[:N]; Xdf=pipe["X_te_r"].iloc[:N].reset_index(drop=True)
            cls_c=st.selectbox("Class",CLASS_NAMES,index=2)
            ci=CLASS_NAMES.index(cls_c)
            with st.spinner("Computing SHAP…"):
                try:
                    ex=shap.TreeExplainer(xgb_mdl)
                    raw=ex.shap_values(Xs)
                    sv=(raw[:,:,ci] if (isinstance(raw,np.ndarray) and raw.ndim==3) else raw[ci])
                    fig8,ax8=plt.subplots(figsize=(9,5),facecolor=DARK_BG)
                    ax8.set_facecolor(DARK_BG); plt.sca(ax8)
                    shap.summary_plot(sv,Xdf,plot_type="dot",max_display=14,show=False,color_bar=True)
                    ax8.set_title(f"SHAP - {cls_c}",color="#e8e8e8"); ax8.tick_params(colors="#888")
                    plt.tight_layout(); st.pyplot(fig8); plt.close()
                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def page_predict(pipe):
    st.title("🔮 Live Credit Score Predictor")
    st.markdown("Enter financial details for an instant prediction with full visual analysis and PDF report.")
    st.markdown("---")
 
    ca,cb,cc = st.columns(3)
    with ca:
        st.markdown("**Personal & Account**")
        age      = st.slider("Age", 18, 75, 35)
        occ      = st.selectbox("Occupation",["Scientist","Teacher","Engineer","Entrepreneur",
                     "Developer","Lawyer","Doctor","Journalist","Manager","Accountant",
                     "Musician","Mechanic","Writer","Architect","Media_Manager"])
        nba      = st.slider("Bank Accounts", 1, 10, 3)
        ncc      = st.slider("Credit Cards", 0, 12, 3)
        nl       = st.slider("Active Loans", 0, 9, 2)
    with cb:
        st.markdown("**Income & Debt**")
        ai       = st.number_input("Annual Income ($)",     5000, 500000, 60000, 1000)
        od       = st.number_input("Outstanding Debt ($)",  0,    5000,   800,   50)
        emi      = st.number_input("Monthly EMI ($)",       0,    5000,   250,   10)
        inv      = st.number_input("Monthly Investment ($)",0,    5000,   300,   50)
        bal      = st.number_input("Monthly Balance ($)",   0,    2000,   400,   50)
    with cc:
        st.markdown("**Credit Behaviour**")
        ir       = st.slider("Interest Rate (%)", 1.0, 34.0, 12.0, 0.5)
        dfd      = st.slider("Avg Delay (days)",  0, 62, 5)
        ndp      = st.slider("Delayed Payments",  0, 22, 2)
        cu       = st.slider("Credit Utilisation (%)", 20.0, 50.0, 30.0, 0.5)
        cha      = st.slider("Credit History (yrs)", 0.0, 35.0, 8.0, 0.5)
        cm       = st.selectbox("Credit Mix", ["Good","Standard","Bad"])
        pma      = st.selectbox("Pays Min Amount?", ["Yes","No","NM"])
        nci      = st.slider("Credit Inquiries", 0, 17, 3)
        ccl      = st.slider("Changed Credit Limit (%)", 0.0, 30.0, 5.0, 0.5)
 
    st.markdown("---")
    if not st.button("🔮  Predict My Credit Score", use_container_width=False):
        return
 
    inp = {"Age":age,"Occupation":occ,"Annual_Income":ai,"Monthly_Inhand_Salary":ai/12,
           "Num_Bank_Accounts":nba,"Num_Credit_Card":ncc,"Interest_Rate":ir,
           "Num_of_Loan":nl,"Delay_from_due_date":dfd,"Num_of_Delayed_Payment":ndp,
           "Changed_Credit_Limit":ccl,"Num_Credit_Inquiries":nci,"Credit_Mix":cm,
           "Outstanding_Debt":od,"Credit_Utilization_Ratio":cu,"Credit_History_Age":cha,
           "Payment_of_Min_Amount":pma,"Total_EMI_per_month":emi,
           "Amount_invested_monthly":inv,"Monthly_Balance":bal}
 
    # Preprocess single row
    raw = pd.DataFrame([inp])
    raw = linearise(raw)
    raw = rebin(raw)
    for col, le in pipe["le_map"].items():
        if col in raw.columns:
            val = raw[col].astype(str).iloc[0]
            if val in le.classes_:
                raw[col] = le.transform([val])
            else:
                raw[col] = le.transform([le.classes_[0]])
    raw_imp = pd.DataFrame(pipe["imputer"].transform(
        raw[[c for c in raw.columns if c!="Credit_Score"]]),
        columns=[c for c in raw.columns if c!="Credit_Score"])
    raw_fe  = engineer(raw_imp)
    sel     = pipe["selected"]
    for c in sel:
        if c not in raw_fe.columns: raw_fe[c] = 0
    raw_sel = pipe["scaler"].transform(raw_fe[raw_fe.columns[raw_fe.columns.isin(
        pipe["X_tr_r"].columns)].tolist()])
    # align columns
    raw_sel_df = pd.DataFrame(raw_sel, columns=raw_fe.columns[raw_fe.columns.isin(
        pipe["X_tr_r"].columns)])
    for c in sel:
        if c not in raw_sel_df.columns: raw_sel_df[c] = 0
    raw_final = raw_sel_df[sel].values
 
    best_mdl = pipe["results"][pipe["best_name"]]["model"]
    pred     = int(best_mdl.predict(raw_final)[0])
    proba    = best_mdl.predict_proba(raw_final)[0]
    score    = proba_to_score(proba)
    lo, hi   = expected_range(ai)
    label    = CLASS_NAMES[pred]
    badge    = {"Poor":"badge-poor","Standard":"badge-standard","Good":"badge-good"}[label]
 
    # ── Row 1: badge + gauge + radar ─────────────────────────────────────────
    r1,r2,r3 = st.columns([1,2,2])
    with r1:
        st.markdown("### Your Score")
        st.markdown(f'<div class="score-badge {badge}">{label}</div>', unsafe_allow_html=True)
        st.markdown(f"**Numeric:** `{score}` / 850")
        st.markdown(f"**Expected:** `{lo}` – `{hi}`")
        st.markdown("**Confidence:**")
        for cn,p in zip(CLASS_NAMES,proba):
            bar="█"*int(p*18)+"░"*(18-int(p*18))
            e={"Poor":"🔴","Standard":"🟡","Good":"🟢"}[cn]
            st.markdown(f"`{e} {cn}` {bar} `{p:.1%}`")
    with r2:
        st.markdown("### Score Gauge")
        st.plotly_chart(gauge(score,lo,hi,f"{label} · {score}/850"), use_container_width=True)
    with r3:
        st.markdown("### Risk Radar")
        st.plotly_chart(radar(radar_dims({**inp,"Credit_Mix":cm,
                                          "Monthly_Inhand_Salary":ai/12})), use_container_width=True)
 
    st.markdown("---")
    # ── Row 2: deviation + SHAP ──────────────────────────────────────────────
    d1,d2 = st.columns(2)
    with d1:
        st.markdown("### Deviation from Good Profile")
        raw_fe_dict = raw_fe.iloc[0].to_dict() if hasattr(raw_fe,"iloc") else {}
        raw_fe_dict.update({"Annual_Income":ai,"Outstanding_Debt":od,
                             "Delay_from_due_date":dfd,"Credit_History_Age":cha,
                             "Credit_Utilization_Ratio":cu,
                             "Num_of_Delayed_Payment":ndp})
        st.plotly_chart(deviation_chart(raw_fe_dict,pipe["good_profile"],pipe["poor_profile"]),
                        use_container_width=True)
        st.markdown("""<div class="info-box">
        🟢 Green = closer to <b>Good</b> class average &nbsp;|&nbsp;
        🔴 Red = further from <b>Good</b> class average</div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("### SHAP - Why this prediction?")
        xgb_mdl = pipe["base"].get("XGBoost (tuned)", pipe["base"].get("LightGBM"))
        wfig = shap_waterfall(xgb_mdl, raw_final, sel, pred)
        if wfig: st.plotly_chart(wfig, use_container_width=True)
        else:    st.info("SHAP waterfall unavailable for this model.")
 
    st.markdown("---")
    # ── Tips ─────────────────────────────────────────────────────────────────
    st.markdown("### 💡 Personalised Insights")
    tips = []
    if dfd > 20:
        tips.append("⚠️ **High payment delays** - paying on or before due date is the single biggest positive action.")
    if od/(ai+1) > 0.3:
        tips.append("⚠️ **Elevated debt-to-income ratio** - reducing outstanding debt will directly improve your score.")
    if cu > 40:
        tips.append("⚠️ **Credit utilisation above 40%** - aim to keep it below 30% of your credit limit.")
    if cha < 3:
        tips.append("💡 **Short credit history** - maintaining existing accounts for longer builds a stronger track record.")
    if cm == "Bad":
        tips.append("💡 **Poor credit mix** - having a diverse combination of credit types signals lower risk to lenders.")
    if ndp > 10:
        tips.append("⚠️ **Many delayed payments on record** - set up automatic payments to prevent future delays.")
    if nci > 8:
        tips.append("⚠️ **High credit inquiry count** - multiple recent inquiries signal credit-seeking behaviour to lenders.")
    if score >= hi:
        tips.append(f"✅ **Above expected range ({lo}–{hi}) for your income bracket** - excellent financial discipline!")
    elif score < lo:
        tips.append(f"💡 **Below expected range ({lo}–{hi}) for your income bracket** - focus on the items flagged above.")
    if not tips:
        tips.append("✅ Your credit profile looks strong. Maintain consistent payment habits to stay here.")
 
    for tip in tips:
        cls = "warn-box" if "⚠️" in tip else "good-box" if "✅" in tip else "info-box"
        st.markdown(f'<div class="{cls}">{tip}</div>', unsafe_allow_html=True)
 
    st.markdown("---")
    # ── PDF Export ───────────────────────────────────────────────────────────
    st.markdown("### 📄 Export Report")
    with st.spinner("Building PDF…"):
        pdf_bytes = build_pdf(inp, score, lo, hi, label, proba, tips,
                              radar_dims({**inp,"Credit_Mix":cm,"Monthly_Inhand_Salary":ai/12}),
                              raw_fe_dict, pipe["good_profile"], pipe["poor_profile"])
    st.download_button("⬇️  Download PDF Report", data=pdf_bytes,
        file_name=f"creditiq_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.get("authenticated"):
        login_page(); return
 
    page = sidebar()
    df   = st.session_state.get("df_uploaded", generate_dataset())
    key  = hashlib.md5(str(df.shape).encode() + str(df.iloc[0].values).encode()).hexdigest()
    n_t  = st.session_state.get("n_trials", 25)
 
    with st.spinner("🔧 Running ML pipeline - SMOTE · RFECV · Optuna · 11 models (cached after first run)…"):
        pipe = run_pipeline(df, key, n_t)
 
    if   page == "🏠 Overview":      page_overview(df, pipe)
    elif page == "📊 EDA":           page_eda(df)
    elif page == "🤖 Models":        page_models(pipe)
    elif page == "🔮 Predict Score": page_predict(pipe)
 
if __name__ == "__main__":
    main()