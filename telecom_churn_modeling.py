# ============================================================
# TELCO CHURN PORTFOLIO: end-to-end script 
# Load -> Inspect -> EDA -> Clean/Encode -> Save Clean -> Collinearity/VIF
# -> Model (imbalance-aware) -> Threshold optimization (cost-based)
# -> Probability calibration + calibration plot
# -> Test evaluation (0.50 + optimized threshold)
# -> Explainability: permutation importance + partial dependence
# -> "Top-risk list" artifact (highest churn probabilities)
#
# - Handle class imbalance using oversampling inside the training pipeline.
# - Model selection uses PR-AUC (average precision), appropriate for imbalance.
# ============================================================

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib

# Imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================
# Paths / project structure
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR  = PROJECT_ROOT / "data"
RAW_PATH  = DATA_DIR / "Telco_customer_churn.xlsx"   

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR    = OUTPUT_DIR / "figures"
MODEL_DIR  = OUTPUT_DIR / "models"
TABLE_DIR  = OUTPUT_DIR / "tables"
CLEAN_DIR  = OUTPUT_DIR / "data_clean"

for d in [DATA_DIR, OUTPUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR, CLEAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Raw file exists:", RAW_PATH.exists(), "|", RAW_PATH.name)

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {RAW_PATH}.\n"
        "Put Telco_customer_churn.xlsx inside the repo's /data folder."
    )

# Optional: reproducibility
RANDOM_STATE = 42

# ============================================================
# BUSINESS GOAL SETTINGS
# ============================================================
# Missing a churner (FN) is usually more expensive than contacting a non-churner (FP).
COST_FN = 5.0   # cost of missing a churner
COST_FP = 1.0   # cost of unnecessary retention outreach

# Threshold grid to search on validation split
THRESHOLDS = np.linspace(0.05, 0.95, 181)  # step ~0.005

# Calibration settings
CALIBRATION_METHOD = "isotonic"  # "sigmoid" or "isotonic"
CALIBRATION_BINS = 10

# Top-risk artifact settings
TOP_RISK_N = 50

# ----------------------------
# Create helper functions
# ----------------------------
def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def to_bool01(series):
    """Map common yes/no/true/false to 1/0. Leaves numeric already-01 as is."""
    if series.dtype != "object":
        return series
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "y": 1, "n": 0,
        "1": 1, "0": 0,
        "no internet service": 0,
        "no phone service": 0
    }
    mapped = s.map(mapping)
    if mapped.notna().mean() >= 0.80:
        return mapped
    return series

def safe_bar(series, title, xlabel, outpath, topn=15):
    vc = series.value_counts(dropna=False)
    if len(vc) > topn:
        vc = vc.head(topn)
    plt.figure(figsize=(10, 4))
    plt.bar(vc.index.astype(str), vc.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def safe_hist(series, title, xlabel, outpath, bins=30):
    s = pd.to_numeric(series, errors="coerce").dropna()
    plt.figure(figsize=(10, 4))
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def churn_rate_by_group(df, group_col, target_col):
    tmp = df[[group_col, target_col]].dropna()
    if tmp.empty:
        return pd.DataFrame()
    out = tmp.groupby(group_col)[target_col].agg(["mean", "size"]).reset_index()
    out = out.rename(columns={"mean": "churn_rate", "size": "n"}).sort_values("churn_rate", ascending=False)
    return out

def plot_churn_rate_table(rate_df, group_col, outpath, title=None):
    if rate_df.empty:
        print(f"Skipping churn-rate plot for {group_col} (no data).")
        return
    plt.figure(figsize=(10, 4))
    plt.bar(rate_df[group_col].astype(str), rate_df["churn_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Churn rate")
    plt.title(title or f"Churn rate by {group_col}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def save_table(df, outpath):
    df.to_csv(outpath, index=False)
    print("Saved:", outpath)

def corr_heatmap_with_values(corr_df, title, outpath, fontsize=7):
    """Heatmap for correlation matrices."""
    vals = corr_df.values
    plt.figure(figsize=(max(12, 0.45 * corr_df.shape[1]),
                        max(10, 0.45 * corr_df.shape[0])))
    im = plt.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="correlation (r)")
    plt.title(title)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90, fontsize=fontsize)
    plt.yticks(range(len(corr_df.index)), corr_df.index, fontsize=fontsize)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                txt_color = "white" if abs(v) > 0.55 else "black"
                plt.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=fontsize, color=txt_color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.30, pad=0.5)
                )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def heatmap_matrix(values, xlabels, ylabels, title, outpath, cmap="coolwarm", vmin=None, vmax=None, annotate_fmt="{:.2f}"):
    """Generic heatmap."""
    arr = np.asarray(values)
    plt.figure(figsize=(10, 5))
    im = plt.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.xticks(range(len(xlabels)), [str(x) for x in xlabels], rotation=45, ha="right")
    plt.yticks(range(len(ylabels)), [str(y) for y in ylabels])
    plt.title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                plt.text(j, i, annotate_fmt.format(v), ha="center", va="center",
                         color="black", fontsize=9,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.20, pad=0.5))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def compute_vif_df(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xn = X.replace([np.inf, -np.inf], np.nan).copy()
    for c in Xn.columns:
        if Xn[c].isna().any():
            Xn[c] = Xn[c].fillna(Xn[c].median())

    arr = Xn.values
    vifs = []
    for i, col in enumerate(Xn.columns):
        vifs.append((col, float(variance_inflation_factor(arr, i))))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

def plot_confusion_matrix_counts_and_rowpct(cm, title, outpath, class_names=("No Churn","Churn")):
    """Confusion matrix heatmap"""
    cm = np.asarray(cm)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, aspect="auto", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.title(title)
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for (i, j), val in np.ndenumerate(cm):
        pct = cm_pct[i, j] * 100 if cm.sum(axis=1)[i] > 0 else 0.0
        txt_color = "white" if im.norm(val) > 0.5 else "black"
        plt.text(
            j, i, f"{val}\n({pct:.1f}%)",
            ha="center", va="center",
            color=txt_color, fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def get_proba_from_pipe(pipe, X_any):
    model_step = pipe.named_steps["model"]
    if hasattr(model_step, "predict_proba"):
        return pipe.predict_proba(X_any)[:, 1]
    if hasattr(model_step, "decision_function"):
        scores = pipe.decision_function(X_any)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    preds = pipe.predict(X_any)
    return preds.astype(float)

def metrics_at_threshold(y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    out = {
        "threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "expected_cost": float(COST_FN * fn + COST_FP * fp),
    }
    return out, cm

def threshold_sweep_cost(y_true, y_proba, thresholds, cost_fn=5.0, cost_fp=1.0):
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = cost_fn * fn + cost_fp * fp
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        rows.append({
            "threshold": float(t),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1v),
            "expected_cost": float(cost),
            "fn_rate": float(fn / (fn + tp + 1e-12)),
            "fp_rate": float(fp / (fp + tn + 1e-12)),
        })
    df = pd.DataFrame(rows).sort_values("expected_cost", ascending=True)
    best_thr = float(df.iloc[0]["threshold"])
    return df, best_thr

def save_cost_curve(df_sweep_sorted_by_threshold, title, outpath):
    plt.figure(figsize=(10, 4))
    plt.plot(df_sweep_sorted_by_threshold["threshold"], df_sweep_sorted_by_threshold["expected_cost"])
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost (C_FN*FN + C_FP*FP)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def calibration_plot(y_true, y_proba, title, outpath, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    print("Saved:", outpath)

def make_calibrated_estimator(prefit_pipe, X_cal, y_cal, method="isotonic"):
    """
    Calibrate a *prefit* pipeline
    """
    try:
        cal = CalibratedClassifierCV(estimator=prefit_pipe, cv="prefit", method=method)
        cal.fit(X_cal, y_cal)
        return cal
    except Exception:
        # newer sklearn prefers FrozenEstimator
        try:
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(estimator=FrozenEstimator(prefit_pipe), method=method)
            cal.fit(X_cal, y_cal)
            return cal
        except Exception as e:
            raise RuntimeError(
                "Calibration failed due to sklearn version differences. "
            ) from e

# ============================================================
# Load data
# ============================================================
df_raw = pd.read_excel(RAW_PATH)
print("\n--- RAW DATA LOADED ---")
print("Shape:", df_raw.shape)
print("Columns:", list(df_raw.columns))

print("\n--- HEAD (10) ---")
try:
    display(df_raw.head(10))
except NameError:
    print(df_raw.head(10))

print("\n--- INFO ---")
df_raw.info()

print("\n--- DESCRIBE (numeric) ---")
try:
    display(df_raw.describe(include=[np.number]).T)
except NameError:
    print(df_raw.describe(include=[np.number]).T)

print("\n--- DESCRIBE (object) ---")
try:
    display(df_raw.describe(include=["object"]).T)
except NameError:
    print(df_raw.describe(include=["object"]).T)

# ============================================================
# Basic checks: NaNs + duplicates
# ============================================================
print("\n--- NaNs summary (top 25) ---")
missing = df_raw.isna().sum().sort_values(ascending=False)
try:
    display(missing.head(25))
except NameError:
    print(missing.head(25))

dup_count = df_raw.duplicated().sum()
print("\nDuplicate rows:", dup_count)

# ============================================================
# Prep for EDA
# ============================================================
df_eda = df_raw.copy()

for c in df_eda.select_dtypes(include=["object"]).columns:
    df_eda[c] = df_eda[c].astype(str).str.strip()
    df_eda.loc[df_eda[c].str.lower().isin(["nan", "none", "null", ""]), c] = np.nan

cols = df_eda.columns

TARGET_CANDIDATES = ["Churn Label", "Churn", "Churn Value"]
TENURE_CANDIDATES = ["tenure", "Tenure Months"]
MONTHLY_CANDIDATES = ["MonthlyCharges", "Monthly Charges"]
TOTAL_CANDIDATES = ["TotalCharges", "Total Charges"]
GENDER_CANDIDATES = ["gender", "Gender"]
CONTRACT_CANDIDATES = ["Contract", "contract"]
INTERNET_CANDIDATES = ["InternetService", "Internet Service"]
PAYMENT_CANDIDATES = ["PaymentMethod", "Payment Method"]
ID_CANDIDATES = ["customerID", "CustomerID", "customer_id", "ID", "Id"]

target_col = first_existing(cols, TARGET_CANDIDATES)
tenure_col = first_existing(cols, TENURE_CANDIDATES)
monthly_col = first_existing(cols, MONTHLY_CANDIDATES)
total_col = first_existing(cols, TOTAL_CANDIDATES)
gender_col = first_existing(cols, GENDER_CANDIDATES)
contract_col = first_existing(cols, CONTRACT_CANDIDATES)
internet_col = first_existing(cols, INTERNET_CANDIDATES)
payment_col = first_existing(cols, PAYMENT_CANDIDATES)
id_col = first_existing(cols, ID_CANDIDATES)

if target_col is None:
    raise ValueError(f"Could not find target column among {TARGET_CANDIDATES}. Found columns: {list(cols)}")

if total_col is not None:
    df_eda[total_col] = pd.to_numeric(df_eda[total_col], errors="coerce")

if df_eda[target_col].dtype == "object":
    y_tmp = df_eda[target_col].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
else:
    y_tmp = pd.to_numeric(df_eda[target_col], errors="coerce")

df_eda["_Churn01"] = y_tmp

print("\n--- Target distribution (raw) ---")
try:
    display(df_eda[target_col].value_counts(dropna=False))
    display(df_eda["_Churn01"].value_counts(dropna=False))
except NameError:
    print(df_eda[target_col].value_counts(dropna=False))
    print(df_eda["_Churn01"].value_counts(dropna=False))


# ============================================================
# Initial EDA plots
# ============================================================
print("\n--- EDA PLOTS ---")

safe_bar(
    df_eda[target_col].fillna("NA"),
    title="Churn Distribution (Raw Labels)",
    xlabel=target_col,
    outpath=FIG_DIR / "eda_churn_distribution.png"
)

if tenure_col is not None:
    safe_hist(df_eda[tenure_col], "Tenure Distribution", tenure_col, FIG_DIR / "eda_tenure_hist.png", bins=30)

if monthly_col is not None:
    safe_hist(df_eda[monthly_col], "Monthly Charges Distribution", monthly_col, FIG_DIR / "eda_monthly_charges_hist.png", bins=30)

if contract_col is not None:
    safe_bar(df_eda[contract_col].fillna("NA"), "Contract Type Counts", contract_col, FIG_DIR / "eda_contract_counts.png")

if internet_col is not None:
    safe_bar(df_eda[internet_col].fillna("NA"), "Internet Service Type Counts", internet_col, FIG_DIR / "eda_internet_service_counts.png")

if payment_col is not None:
    safe_bar(df_eda[payment_col].fillna("NA"), "Payment Method Counts", payment_col, FIG_DIR / "eda_payment_method_counts.png")

if gender_col is not None:
    rate = churn_rate_by_group(df_eda, gender_col, "_Churn01")
    save_table(rate, TABLE_DIR / "eda_churn_rate_by_gender.csv")
    plot_churn_rate_table(rate, gender_col, FIG_DIR / "eda_churn_rate_by_gender.png")

if contract_col is not None:
    rate = churn_rate_by_group(df_eda, contract_col, "_Churn01")
    save_table(rate, TABLE_DIR / "eda_churn_rate_by_contract.csv")
    plot_churn_rate_table(rate, contract_col, FIG_DIR / "eda_churn_rate_by_contract.png")

if internet_col is not None:
    rate = churn_rate_by_group(df_eda, internet_col, "_Churn01")
    save_table(rate, TABLE_DIR / "eda_churn_rate_by_internet.csv")
    plot_churn_rate_table(rate, internet_col, FIG_DIR / "eda_churn_rate_by_internet.png")

if payment_col is not None:
    rate = churn_rate_by_group(df_eda, payment_col, "_Churn01")
    save_table(rate, TABLE_DIR / "eda_churn_rate_by_payment.csv")
    plot_churn_rate_table(rate, payment_col, FIG_DIR / "eda_churn_rate_by_payment.png")

# heatmap: Payment Method × Internet Service 
if (payment_col is not None) and (internet_col is not None):
    tmp = df_eda[[payment_col, internet_col, "_Churn01"]].dropna()
    if not tmp.empty:
        pivot = tmp.pivot_table(index=payment_col, columns=internet_col, values="_Churn01", aggfunc="mean")
        vmin, vmax = float(np.nanmin(pivot.values)), float(np.nanmax(pivot.values))
        outp = FIG_DIR / "eda_churn_heatmap_payment_x_internet.png"
        heatmap_matrix(
            pivot.values,
            xlabels=pivot.columns.astype(str).tolist(),
            ylabels=pivot.index.astype(str).tolist(),
            title="Churn Rate Heatmap: Payment Method × Internet Service",
            outpath=outp,
            cmap="coolwarm",
            vmin=vmin, vmax=vmax,
            annotate_fmt="{:.2f}"
        )

# ============================================================
# CLEANING + ENCODING (keep ID separately for artifacts)
# ============================================================
print("\n--- CLEANING + ENCODING ---")

df = df_raw.copy()

if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()
    df.loc[df[c].str.lower().isin(["nan", "none", "null", ""]), c] = np.nan

cols = df.columns
target_col = first_existing(cols, TARGET_CANDIDATES)
tenure_col = first_existing(cols, TENURE_CANDIDATES)
monthly_col = first_existing(cols, MONTHLY_CANDIDATES)
total_col = first_existing(cols, TOTAL_CANDIDATES)
gender_col = first_existing(cols, GENDER_CANDIDATES)
contract_col = first_existing(cols, CONTRACT_CANDIDATES)
internet_col = first_existing(cols, INTERNET_CANDIDATES)
payment_col = first_existing(cols, PAYMENT_CANDIDATES)
id_col = first_existing(cols, ID_CANDIDATES)

# Keep an ID column for "top risk list" if present; otherwise synthesize one
if id_col is not None:
    customer_id = df[id_col].astype(str)
else:
    customer_id = pd.Series([f"row_{i}" for i in range(len(df))], name="CustomerID")

if df[target_col].dtype == "object":
    y = df[target_col].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
else:
    y = pd.to_numeric(df[target_col], errors="coerce")

if y.isna().any():
    raise ValueError(
        f"Could not map target {target_col} cleanly to 0/1. "
        f"Unique values: {sorted(df[target_col].dropna().unique().tolist())}"
    )

df["Churn Value"] = y.astype(int)

# Numeric handling
if total_col is not None:
    df[total_col] = df[total_col].replace(" ", np.nan)
    df["Total Charges"] = pd.to_numeric(df[total_col], errors="coerce")
else:
    df["Total Charges"] = np.nan

if tenure_col is not None:
    df["Tenure Months"] = pd.to_numeric(df[tenure_col], errors="coerce")
else:
    df["Tenure Months"] = np.nan

if monthly_col is not None:
    df["Monthly Charges"] = pd.to_numeric(df[monthly_col], errors="coerce")
else:
    df["Monthly Charges"] = np.nan

if gender_col is not None:
    g = df[gender_col].astype(str).str.strip().str.lower()
    df["Gender"] = g.map({"male": 1, "female": 0})
else:
    df["Gender"] = np.nan

bin_like = {
    "Senior Citizen": ["SeniorCitizen", "Senior Citizen"],
    "Partner": ["Partner"],
    "Dependents": ["Dependents"],
    "Phone Service": ["PhoneService", "Phone Service"],
    "Multiple Lines": ["MultipleLines", "Multiple Lines"],
    "Tech Support": ["TechSupport", "Tech Support"],
    "Paperless Billing": ["PaperlessBilling", "Paperless Billing"],
    "Online Security": ["OnlineSecurity", "Online Security"],
    "Online Backup": ["OnlineBackup", "Online Backup"],
    "Device Protection": ["DeviceProtection", "Device Protection"],
    "Streaming TV": ["StreamingTV", "Streaming TV"],
    "Streaming Movies": ["StreamingMovies", "Streaming Movies"],
}

for out_col, cands in bin_like.items():
    src = first_existing(cols, cands)
    if src is None:
        df[out_col] = np.nan
        continue
    s = df[src]
    if s.dtype == "object":
        df[out_col] = to_bool01(s)
    else:
        df[out_col] = pd.to_numeric(s, errors="coerce")

if internet_col is None:
    df["Internet Service"] = np.nan
else:
    df["Internet Service"] = df[internet_col].astype(str).str.strip()

if contract_col is None:
    df["Contract"] = np.nan
else:
    df["Contract"] = df[contract_col].astype(str).str.strip()

if payment_col is None:
    df["Payment Method"] = np.nan
else:
    df["Payment Method"] = df[payment_col].astype(str).str.strip()

# Keep a table for top-risk artifact (pre one-hot)
df_human = pd.DataFrame({
    "CustomerID": customer_id.values,
    "Tenure Months": df["Tenure Months"].values,
    "Monthly Charges": df["Monthly Charges"].values,
    "Contract": df["Contract"].values,
    "Internet Service": df["Internet Service"].values,
    "Payment Method": df["Payment Method"].values,
    "Tech Support": df.get("Tech Support", pd.Series([np.nan]*len(df))).values,
    "Online Security": df.get("Online Security", pd.Series([np.nan]*len(df))).values,
    "Paperless Billing": df.get("Paperless Billing", pd.Series([np.nan]*len(df))).values,
    "Dependents": df.get("Dependents", pd.Series([np.nan]*len(df))).values,
    "Churn Value": df["Churn Value"].values
})

# Modeling columns
keep_cols = [
    "Gender","Senior Citizen","Partner","Dependents","Tenure Months",
    "Phone Service","Multiple Lines","Tech Support",
    "Internet Service","Online Security","Online Backup","Device Protection",
    "Streaming TV","Streaming Movies",
    "Contract","Paperless Billing","Payment Method",
    "Monthly Charges","Total Charges",
    "Churn Value"
]
for c in keep_cols:
    if c not in df.columns:
        df[c] = np.nan

df_clean = df[keep_cols].copy()

for c in ["Tenure Months", "Monthly Charges", "Total Charges"]:
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    df_clean[c] = df_clean[c].fillna(df_clean[c].median())

binary_out_cols = [
    "Gender","Senior Citizen","Partner","Dependents","Phone Service","Multiple Lines","Tech Support",
    "Online Security","Online Backup","Device Protection","Streaming TV","Streaming Movies",
    "Paperless Billing","Churn Value"
]
for c in binary_out_cols:
    if df_clean[c].isna().any():
        mode_val = df_clean[c].mode(dropna=True)
        fill = float(mode_val.iloc[0]) if len(mode_val) else 0.0
        df_clean[c] = df_clean[c].fillna(fill)
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0).astype(int)

# One-hot
internet_dum = pd.get_dummies(df_clean["Internet Service"], prefix="Internet Service", dtype=int)
contract_dum = pd.get_dummies(df_clean["Contract"], prefix="Contract", dtype=int)
pay_dum = pd.get_dummies(df_clean["Payment Method"], dtype=int)

desired_pay_cols = [
    "Bank transfer (automatic)", "Credit card (automatic)", "Mailed check", "Electronic check"
]
for c in desired_pay_cols:
    if c not in pay_dum.columns:
        pay_dum[c] = 0
pay_dum = pay_dum[desired_pay_cols].astype(int)

df_final = df_clean.drop(columns=["Internet Service","Contract","Payment Method"])
df_final = pd.concat([df_final, internet_dum, contract_dum, pay_dum], axis=1)

bool_cols = df_final.select_dtypes(include=["bool"]).columns
df_final[bool_cols] = df_final[bool_cols].astype(int)

# Save cleaned data
clean_csv = CLEAN_DIR / "final_cleaned_data.csv"
clean_xlsx = CLEAN_DIR / "final_cleaned_data.xlsx"
df_final.to_csv(clean_csv, index=False)
df_final.to_excel(clean_xlsx, index=False)
print("Saved clean:", clean_csv)
print("Saved clean:", clean_xlsx)
print("Final clean shape:", df_final.shape)


# ============================================================
# COLLINEARITY CHECK + VIF = TO GET THE FINAL PREDICTOR SET
# ============================================================
print("\n--- COLLINEARITY + VIF (feature reduction) ---")

TARGET = "Churn Value"
X_full = df_final.drop(columns=[TARGET]).copy()
y_full = df_final[TARGET].astype(int).copy()

corr0 = X_full.corr()
corr0_path = FIG_DIR / "corr_heatmap_model_input.png"
corr_heatmap_with_values(corr0, "Correlation Heatmap (Model Input Features)", corr0_path)

# Drop ONE dummy per group (before VIF) to avoid perfect multicollinearity
DROP_BEFORE_VIF = [
    "Internet Service_No",
    "Contract_Month-to-month",
    "Electronic check"
]
X_vif = X_full.drop(columns=[c for c in DROP_BEFORE_VIF if c in X_full.columns])

vif0 = compute_vif_df(X_vif)
vif0_path = TABLE_DIR / "vif_initial.csv"
vif0.to_csv(vif0_path, index=False)
print("Saved:", vif0_path)

# Drop Total Charges (structural dependence)
if "Total Charges" in X_vif.columns:
    X_vif = X_vif.drop(columns=["Total Charges"])

vif1 = compute_vif_df(X_vif)
vif1_path = TABLE_DIR / "vif_after_drop_total_charges.csv"
vif1.to_csv(vif1_path, index=False)
print("Saved:", vif1_path)

X_final = X_vif.copy()
y_final = y_full.copy()

df_model = pd.concat([X_final, y_final], axis=1)
model_csv = CLEAN_DIR / "final_modeling_data.csv"
df_model.to_csv(model_csv, index=False)
print("Saved:", model_csv)

corr1 = X_final.corr()
corr1_path = FIG_DIR / "corr_heatmap_after_vif.png"
corr_heatmap_with_values(corr1, "Correlation Heatmap (After VIF Reduction)", corr1_path)


# ============================================================
# MODELING (imbalance-aware) + THRESHOLD OPTIMIZATION
# ============================================================
print("\n--- MODELING (imbalance-aware) ---")

X = X_final.copy()
y = y_final.copy()

print("\nClass balance (full data):")
print(y.value_counts())
print("Churn rate:", y.mean())

# Make sure df_human aligns with X/y rows
df_human = df_human.loc[X.index].reset_index(drop=True)

# Train/Test split (keep human rows aligned)
X_train, X_test, y_train, y_test, human_train, human_test = train_test_split(
    X, y, df_human,
    test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

numeric_features = X.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ]), numeric_features)
    ],
    remainder="drop"
)

# Candidate models
models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
    "HistGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "roc_auc": "roc_auc",
    "avg_precision": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

def make_training_pipe(clf):
    # Oversampling occurs inside CV folds (no leakage). Keep it internal.
    return ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("oversample", SMOTE(random_state=RANDOM_STATE)),
        ("model", clf)
    ])

def run_cv_and_select_best(models_dict):
    rows = []
    for name, clf in models_dict.items():
        pipe = make_training_pipe(clf)
        res = cross_validate(
            pipe, X_train, y_train,
            cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False
        )
        row = {"model": name}
        for k, v in res.items():
            if k.startswith("test_"):
                row[k.replace("test_", "")] = float(np.mean(v))
                row[k.replace("test_", "") + "_std"] = float(np.std(v))
        rows.append(row)

    # Select best by PR-AUC (avg_precision)
    cv_df = pd.DataFrame(rows).sort_values("avg_precision", ascending=False)
    out_csv = TABLE_DIR / "cv_model_comparison.csv"
    cv_df.to_csv(out_csv, index=False)
    print("\nCV results saved:", out_csv)
    try:
        display(cv_df)
    except NameError:
        print(cv_df)

    best_name = cv_df.iloc[0]["model"]
    return cv_df, best_name

cv_df, best_name = run_cv_and_select_best(models)
print("\nSelected model:", best_name)

# Fit chosen model on full training split (for final test evaluation later)
best_pipe = make_training_pipe(models[best_name])
best_pipe.fit(X_train, y_train)


# ============================================================
# Threshold optimization + calibration (validation split from train)
# ============================================================
print("\n--- VALIDATION: Threshold optimization + probability calibration ---")

# Split training into fit/calibration/validation:
# - Fit: train the model
# - Cal: calibrate probabilities
# - Val: choose operating threshold by expected business cost
X_fit, X_tmp, y_fit, y_tmp, human_fit, human_tmp = train_test_split(
    X_train, y_train, human_train,
    test_size=0.40, random_state=RANDOM_STATE, stratify=y_train
)
X_cal, X_val, y_cal, y_val, human_cal, human_val = train_test_split(
    X_tmp, y_tmp, human_tmp,
    test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp
)

# (a) Fit model on X_fit
pipe_for_cal = make_training_pipe(models[best_name])
pipe_for_cal.fit(X_fit, y_fit)

# (b) Calibrate on X_cal (prefit calibration)
calibrated = make_calibrated_estimator(
    prefit_pipe=pipe_for_cal,
    X_cal=X_cal,
    y_cal=y_cal,
    method=CALIBRATION_METHOD
)

# Calibration plot on calibration split
cal_proba = calibrated.predict_proba(X_cal)[:, 1]
cal_plot_path = FIG_DIR / "calibration_plot.png"
calibration_plot(
    y_true=y_cal.values,
    y_proba=cal_proba,
    title=f"Calibration Plot ({CALIBRATION_METHOD}) - calibration split",
    outpath=cal_plot_path,
    n_bins=CALIBRATION_BINS
)

# (c) Threshold sweep on validation split (use calibrated probabilities)
val_proba = calibrated.predict_proba(X_val)[:, 1]
sweep_df, best_thr = threshold_sweep_cost(
    y_true=y_val.values,
    y_proba=val_proba,
    thresholds=THRESHOLDS,
    cost_fn=COST_FN,
    cost_fp=COST_FP
)

sweep_out = TABLE_DIR / "threshold_sweep.csv"
sweep_df.to_csv(sweep_out, index=False)
print("Saved:", sweep_out)

cost_plot = FIG_DIR / "cost_curve_threshold.png"
save_cost_curve(
    sweep_df.sort_values("threshold"),
    title="Cost vs Threshold  [validation]",
    outpath=cost_plot
)

print(f"Best threshold (min expected cost): {best_thr:.3f}  (C_FN={COST_FN}, C_FP={COST_FP})")


# ============================================================
# TEST EVALUATION (0.50 vs optimized threshold) using calibrated model
# ============================================================
print("\n--- TEST EVALUATION (0.50 vs optimized threshold) ---")

test_proba = calibrated.predict_proba(X_test)[:, 1]

m_05, cm_05 = metrics_at_threshold(y_test.values, test_proba, 0.50)
m_opt, cm_opt = metrics_at_threshold(y_test.values, test_proba, best_thr)

metrics_df = pd.DataFrame([
    {"threshold_policy": "default_0.50", **m_05, "cost_fn": COST_FN, "cost_fp": COST_FP},
    {"threshold_policy": f"optimized_{best_thr:.3f}", **m_opt, "cost_fn": COST_FN, "cost_fp": COST_FP},
])
metrics_out = TABLE_DIR / "test_metrics_threshold_comparison.csv"
metrics_df.to_csv(metrics_out, index=False)
print("Saved:", metrics_out)

try:
    display(metrics_df)
except NameError:
    print(metrics_df)

print("\n=== Classification report (optimized threshold) ===")
pred_opt = (test_proba >= best_thr).astype(int)
print("Model:", best_name, "| threshold:", f"{best_thr:.3f}")
print(classification_report(y_test, pred_opt, target_names=["No Churn", "Churn"]))

plot_confusion_matrix_counts_and_rowpct(
    cm_05,
    title=f"Confusion Matrix ({best_name})  thr=0.50",
    outpath=FIG_DIR / "confusion_matrix_thr050.png"
)
plot_confusion_matrix_counts_and_rowpct(
    cm_opt,
    title=f"Confusion Matrix ({best_name})  thr={best_thr:.3f}",
    outpath=FIG_DIR / "confusion_matrix_optthr.png"
)

# ROC and PR curves (single model)
plt.figure()
RocCurveDisplay.from_predictions(y_test, test_proba, name=f"{best_name}")
plt.title("ROC Curve (Test)")
plt.tight_layout()
roc_path = FIG_DIR / "roc_curve_test.png"
plt.savefig(roc_path, dpi=200)
plt.show()
print("Saved:", roc_path)

plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, test_proba, name=f"{best_name}")
plt.title("Precision-Recall Curve (Test)")
plt.tight_layout()
pr_path = FIG_DIR / "pr_curve_test.png"
plt.savefig(pr_path, dpi=200)
plt.show()
print("Saved:", pr_path)


# ============================================================
# Explainability- Permutation importance (model-agnostic) & 
# Partial dependence 
# ============================================================
print("\n--- EXPLAINABILITY ---")

def save_permutation_importance(calibrated_model, tag="final"):
    # Use calibrated model directly
    perm = permutation_importance(
        calibrated_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    fi_df = pd.DataFrame({
        "feature": X_test.columns.to_numpy(),
        "importance": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance", ascending=False)

    out_csv = TABLE_DIR / f"feature_importance_permutation.csv"
    fi_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    top10 = fi_df.head(10).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top10["feature"], top10["importance"])
    plt.title(f"Top 10 Feature Importance ({best_name})")
    plt.xlabel("Permutation Importance (mean)")
    plt.tight_layout()
    out_png = FIG_DIR / f"feature_importance.png"
    plt.savefig(out_png, dpi=200)
    plt.show()
    print("Saved:", out_png)

    return fi_df

fi_df = save_permutation_importance(calibrated, tag="final")

# Partial Dependence: take top 3 numeric features by permutation importance
top_features = fi_df["feature"].head(3).tolist()
# safety: ensure they exist
top_features = [f for f in top_features if f in X.columns]
if len(top_features) > 0:
    plt.figure(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        calibrated, X_test, features=top_features, kind="average"
    )
    plt.suptitle("Partial Dependence (Top Features)")
    plt.tight_layout()
    pdp_path = FIG_DIR / "partial_dependence_top_features.png"
    plt.savefig(pdp_path, dpi=200)
    plt.show()
    print("Saved:", pdp_path)
else:
    print("Skipping partial dependence (no valid top features found).")


# ============================================================
# Top-risk list artifact (highest churn probabilities)
# ============================================================
print("\n--- TOP-RISK LIST ARTIFACT ---")

risk_df = human_test.copy().reset_index(drop=True)
risk_df["churn_probability"] = test_proba
risk_df["predicted_label_opt"] = (test_proba >= best_thr).astype(int)

# Sort descending by risk
risk_df = risk_df.sort_values("churn_probability", ascending=False)

top_risk = risk_df.head(TOP_RISK_N).copy()
top_risk_out = TABLE_DIR / "top_risk_customers_test.csv"
top_risk.to_csv(top_risk_out, index=False)
print("Saved:", top_risk_out)

# Also save a compact version (put most interpretable columns first)
compact_cols = [
    "CustomerID", "churn_probability", "predicted_label_opt",
    "Tenure Months", "Monthly Charges", "Contract", "Internet Service", "Payment Method",
    "Tech Support", "Online Security", "Paperless Billing", "Dependents"
]
compact_cols = [c for c in compact_cols if c in top_risk.columns]
top_risk_compact_out = TABLE_DIR / "top_risk_customers_test_compact.csv"
top_risk[compact_cols].to_csv(top_risk_compact_out, index=False)
print("Saved:", top_risk_compact_out)


# ============================================================
# Save model artifacts + metadata
# ============================================================
print("\n--- SAVING MODEL + METADATA ---")

model_path = MODEL_DIR / f"{best_name}_pipeline_calibrated.joblib"
joblib.dump(calibrated, model_path)
print("Saved model:", model_path)

meta = {
    "selected_model": best_name,
    "selection_metric_cv": "avg_precision",
    "threshold_opt": float(best_thr),
    "cost_fn": float(COST_FN),
    "cost_fp": float(COST_FP),
    "calibration_method": CALIBRATION_METHOD,
    "calibration_bins": int(CALIBRATION_BINS),
    "notes": (
        "Imbalance handled inside training pipeline"
        "Threshold optimized on validation split using expected cost. "
        "Probabilities calibrated on a separate calibration split."
    )
}
meta_path = TABLE_DIR / "run_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Saved:", meta_path)

print("\nDONE.")

