import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
from pathlib import Path
from io import BytesIO
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance
import joblib

# ---------- Page ----------
st.set_page_config(page_title="ECA Multi-Output RF (J & CTOD)", layout="wide")
st.title("ECA Multi-Output Random Forest: Predict J & CTOD")

with st.expander("About this app", expanded=False):
    st.markdown(
        "- Trains a multi-output Random Forest from 5 inputs to predict 6 J/CTOD targets.\n"
        "- Handles categorical features via one-hot (Sour Region, Notch Location).\n"
        "- Shows per-target metrics, OOB R², aggregated & per-target permutation importance, diagnostics, and a Model Card.\n"
        "- Try a single prediction or batch-predict via upload."
    )

# ---------- Sidebar Controls ----------
st.sidebar.header("Settings")

DEFAULT_FILE = "ECA ML.xlsx"
input_choice = st.sidebar.selectbox(
    "Choose data source", ("Use bundled path", "Upload Excel file (.xlsx)")
)

if input_choice == "Use bundled path":
    input_path = st.sidebar.text_input("Excel path", value=DEFAULT_FILE)
    sheet_name = st.sidebar.text_input("Sheet name", value="ML database")
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])
    sheet_name = st.sidebar.text_input("Sheet name", value="ML database")
    input_path = None

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.sidebar.slider("n_estimators", 200, 1000, 500, 50)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
run_cv = st.sidebar.checkbox("Run 5-fold CV (average R²)", value=False)

st.sidebar.markdown("---")
do_save = st.sidebar.checkbox("Save trained model", value=True)
model_filename = st.sidebar.text_input("Model filename", value="eca_multioutput_rf.joblib")

# ---------- Constants (locked features/targets) ----------
RAW_FEATURES = ["Sour region", "pH", "ppH2S(bara)", "K-rate", "Notch Location"]
CANON_FEATURES = ["Sour Region", "pH", "ppH2S", "K-rate", "Notch Location"]  # after rename
CATEGORICAL_FEATS = ["Sour Region", "Notch Location"]  # treat both as categorical
TARGET_COLS_LOCKED = ["J0", "J0.2", "Jmax", "CTOD0", "CTOD0.2", "CTODmax"]

# ---------- Helpers ----------
def make_ohe():
    # Version-proof OneHotEncoder (sklearn <1.2 uses 'sparse', >=1.2 uses 'sparse_output')
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

@st.cache_data(show_spinner=False)
def load_excel(_uploaded_file, _input_path, _sheet):
    if _uploaded_file is not None:
        df = pd.read_excel(_uploaded_file, sheet_name=_sheet)
    else:
        df = pd.read_excel(_input_path, sheet_name=_sheet)
    # normalize names
    df.columns = [c.strip().replace("\u200b", "") for c in df.columns]
    rename_map = {
        "Sour region": "Sour Region",
        "ppH2S(bara)": "ppH2S",
        "Test Pressure(bara)": "Test Pressure",
        "a/w": "a/W",
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def get_all_feature_names(preprocessor, X_sample, num_cols, cat_cols):
    """
    Returns:
      full_names: names with transformer prefixes like 'num__pH', 'cat__Sour Region_Sour'
      clean_names: stripped names like 'pH' or 'Sour Region_Sour'
    """
    # Try to ask ColumnTransformer directly
    try:
        full_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback: build manually
        full_names = []
        for c in num_cols:
            full_names.append(f"num__{c}")
        if "cat" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["cat"]
            try:
                ohe_names = ohe.get_feature_names_out(cat_cols)
            except Exception:
                # last-resort: generic counts based on observed categories in sample
                ohe_names = []
                for col in cat_cols:
                    n_cat = len(pd.Series(X_sample[col]).dropna().unique())
                    ohe_names.extend([f"{col}_{i}" for i in range(n_cat)])
            full_names.extend([f"cat__{n}" for n in ohe_names])

    clean_names = [n.split("__", 1)[1] if "__" in n else n for n in full_names]

    # Hard check: count must match transformed width
    try:
        transformed_dim = preprocessor.transform(X_sample.iloc[:1]).shape[1]
    except Exception:
        transformed_dim = len(full_names)

    if len(full_names) != transformed_dim:
        # align with width to avoid downstream mismatch
        full_names = [f"feat__{i}" for i in range(transformed_dim)]
        clean_names = full_names[:]

    return full_names, clean_names

@st.cache_resource(show_spinner=True)
def train_model(X, y, num_cols, cat_cols, test_size, n_estimators, random_state):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    pipe = Pipeline([("prep", preprocess), ("model", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe.fit(X_train, y_train)
    return pipe, (X_train, X_test, y_train, y_test)

# ---------- Load Data ----------
try:
    df = load_excel(uploaded_file, input_path, sheet_name)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.subheader("Preview data")
st.dataframe(df.head(20), use_container_width=True)

# ---------- Build feature/target frames (locked) ----------
missing_feats = [c for c in CANON_FEATURES if c not in df.columns]
missing_tgts = [t for t in TARGET_COLS_LOCKED if t not in df.columns]

if missing_feats:
    st.warning(f"Missing required feature columns (after renaming): {missing_feats}")
if missing_tgts:
    st.error(f"Missing required target columns: {missing_tgts}")
    st.stop()

feature_cols = CANON_FEATURES[:]        # locked
target_cols = TARGET_COLS_LOCKED[:]     # locked

# Cat/Num split
cat_cols = [c for c in CATEGORICAL_FEATS if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Basic NA handling
rows_before = len(df)
data = df[feature_cols + target_cols].dropna(axis=0).reset_index(drop=True)
rows_after = len(data)
if rows_after < 10:
    st.warning("Very small dataset after dropping NAs; results may be unstable.")
X = data[feature_cols]
y = data[target_cols]

# ---------- Train ----------
train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pipe, (X_train, X_test, y_train, y_test) = train_model(
    X, y, num_cols, cat_cols, test_size, n_estimators, random_state
)
st.success("Model trained.")

# ---------- Evaluation ----------
y_pred = pipe.predict(X_test)
metrics = []
for i, col in enumerate(target_cols):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    metrics.append({"target": col, "R2": r2, "RMSE": rmse})
metrics_df = pd.DataFrame(metrics).sort_values("target").reset_index(drop=True)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Per-target metrics")
    st.dataframe(metrics_df, use_container_width=True)
with col2:
    st.metric("Avg R²", f"{metrics_df['R2'].mean():.3f}")
    st.metric("Avg RMSE", f"{metrics_df['RMSE'].mean():.3f}")
    oob = getattr(pipe.named_steps["model"], "oob_score_", None)
    if oob is not None:
        st.metric("OOB R²", f"{oob:.3f}")

# Optional CV
if run_cv:
    st.info("Running 5-fold CV on full data (this may take a while).")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_validate(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    st.write(f"CV R² (mean ± std): {scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}")

# ---------- Permutation Importance (aggregated) ----------
with st.spinner("Computing permutation importance (aggregated)..."):
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )

full_names, clean_names = get_all_feature_names(
    pipe.named_steps["prep"], X_test, num_cols, cat_cols
)

imp_mean = result.importances_mean
imp_std  = result.importances_std

# Build base-feature mapping
def base_feature(clean):
    # 'Sour Region_Sour' -> 'Sour Region'
    for c in cat_cols:
        if clean.startswith(c + "_"):
            return c
    return clean

pi_df = pd.DataFrame({
    "full_name": full_names[:len(imp_mean)],   # slice defensively
    "clean_name": clean_names[:len(imp_mean)],
    "importance_mean": imp_mean,
    "importance_std": imp_std,
})
pi_df["base_feature"] = pi_df["clean_name"].apply(base_feature)

agg = (pi_df.groupby("base_feature", as_index=False)["importance_mean"]
          .sum()
          .sort_values("importance_mean", ascending=False))

st.subheader("Permutation importance (aggregated to original features)")
st.dataframe(agg, use_container_width=True)

# Bar chart (single plot, default colors)
fig, ax = plt.subplots(figsize=(8, max(3, len(agg)*0.4)))
ax.barh(agg["base_feature"], agg["importance_mean"])
ax.invert_yaxis()
ax.set_title("Permutation Importance (Aggregated)")
ax.set_xlabel("Mean importance (validation)")
ax.set_ylabel("Feature")
st.pyplot(fig, clear_figure=True)

# ---------- Per-target Permutation Importance ----------
with st.expander("Permutation importance by target", expanded=False):

    # custom scorer that evaluates only the i-th output column from the pipeline
    def r2_single_output(y_true, y_pred, *, idx):
        return r2_score(y_true, y_pred[:, idx])

    for i, tgt in enumerate(target_cols):
        scorer_i = make_scorer(r2_single_output, greater_is_better=True, idx=i)

        with st.spinner(f"Computing permutation importance for {tgt}..."):
            res_t = permutation_importance(
                pipe,
                X_test,
                y_test.iloc[:, i].to_numpy(),   # 1D y for target i
                n_repeats=10,
                random_state=random_state,
                n_jobs=-1,
                scoring=scorer_i,
            )

        _, clean_names_t = get_all_feature_names(
            pipe.named_steps["prep"], X_test, num_cols, cat_cols
        )
        mean_t = res_t.importances_mean

        pi_t = (
            pd.DataFrame({"clean_name": clean_names_t[:len(mean_t)], "importance": mean_t})
            .assign(base_feature=lambda d: d["clean_name"].apply(base_feature))
            .groupby("base_feature", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False)
        )

        st.write(f"**Target:** {tgt}")
        st.dataframe(pi_t, use_container_width=True)

        fig_t, ax_t = plt.subplots(figsize=(8, max(3, len(pi_t)*0.4)))
        ax_t.barh(pi_t["base_feature"], pi_t["importance"])
        ax_t.invert_yaxis()
        ax_t.set_title(f"Permutation Importance — {tgt}")
        ax_t.set_xlabel("Mean importance (validation)")
        ax_t.set_ylabel("Feature")
        st.pyplot(fig_t, clear_figure=True)

# ---------- Diagnostics: Actual vs Predicted ----------
with st.expander("Diagnostics: Actual vs Predicted (per target)", expanded=False):
    for i, col in enumerate(target_cols):
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test.iloc[:, i], y_pred[:, i])
        ax2.set_xlabel(f"Actual {col}")
        ax2.set_ylabel(f"Predicted {col}")
        ax2.set_title(f"Actual vs Predicted — {col}")
        mn = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
        mx = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([mn, mx], [mn, mx])
        st.pyplot(fig2, clear_figure=True)

st.markdown("---")

# ---------- Model Card ----------
with st.expander("Model Card", expanded=True):
    info = {
        "Timestamp": train_time,
        "Data rows (before dropna)": rows_before,
        "Data rows (after dropna)": rows_after,
        "Train/Test Split": f"{1 - test_size:.0%} / {test_size:.0%}",
        "Features (locked)": ", ".join(feature_cols),
        "Categorical": ", ".join(cat_cols) if len(cat_cols) else "(none)",
        "Numeric": ", ".join(num_cols) if len(num_cols) else "(none)",
        "Targets (locked)": ", ".join(target_cols),
        "n_estimators": n_estimators,
        "random_state": random_state,
        "Bootstrap": True,
        "OOB R²": f"{getattr(pipe.named_steps['model'], 'oob_score_', float('nan')):.3f}",
        "CV run?": "Yes" if run_cv else "No",
        "Avg Test R²": f"{metrics_df['R2'].mean():.3f}",
        "Avg Test RMSE": f"{metrics_df['RMSE'].mean():.3f}",
    }
    df_info = pd.DataFrame(list(info.items()), columns=["Field", "Value"])
    st.table(df_info)

st.markdown("---")

# ---------- Save model ----------
if do_save:
    try:
        joblib.dump(pipe, model_filename)
        st.success(f"Saved model to: {model_filename}")
        with open(model_filename, "rb") as f:
            st.download_button("Download model file", f, file_name=model_filename)
    except Exception as e:
        st.warning(f"Could not save model: {e}")

# ---------- Inference (single row) ----------
st.subheader("Try a single prediction")
with st.form("single_pred"):
    inputs = {}
    for col in feature_cols:
        if col in cat_cols:
            cats = sorted(pd.Series(X[col]).dropna().unique().tolist())
            if len(cats) == 0:
                st.warning(f"No categories found in training data for '{col}'. Using empty string.")
                cats = [""]
            inputs[col] = st.selectbox(col, options=cats, index=0)
        else:
            series = pd.Series(X[col]).dropna()
            default_val = float(pd.to_numeric(series, errors="coerce").median()) if len(series) else 0.0
            inputs[col] = st.number_input(col, value=default_val, step=0.001, format="%.6f")
    submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            row = pd.DataFrame([inputs])
            pred = pipe.predict(row)
            pred_df = pd.DataFrame(pred, columns=target_cols)
            st.write("Prediction:")
            st.dataframe(pred_df, use_container_width=True)
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download prediction (CSV)", data=csv, file_name="single_prediction.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- Batch Inference ----------
st.subheader("Batch prediction (upload new rows)")
batch_file = st.file_uploader("Upload CSV or Excel with **feature columns only**", type=["csv","xlsx"], key="batch")
if batch_file is not None:
    try:
        if batch_file.name.lower().endswith(".csv"):
            newX = pd.read_csv(batch_file)
        else:
            newX = pd.read_excel(batch_file)
        st.write("Preview of uploaded features:")
        st.dataframe(newX.head(), use_container_width=True)

        # Precompute training fill values (medians for num, modes for cat)
        num_fills = {c: float(pd.to_numeric(X[c], errors="coerce").median()) if X[c].dropna().size else 0.0
                     for c in num_cols}
        cat_fills = {c: (pd.Series(X[c]).mode().iat[0] if not pd.Series(X[c]).mode().empty else "")
                     for c in cat_cols}

        # Ensure required columns exist & coerce types
        for c in feature_cols:
            if c not in newX.columns:
                newX[c] = cat_fills[c] if c in cat_cols else num_fills.get(c, 0.0)

        for c in num_cols:
            newX[c] = pd.to_numeric(newX[c], errors="coerce")
            newX[c].fillna(num_fills[c], inplace=True)
        for c in cat_cols:
            newX[c] = newX[c].astype(str).fillna(cat_fills[c])

        preds = pipe.predict(newX[feature_cols])
        preds_df = pd.DataFrame(preds, columns=target_cols)
        out = pd.concat([newX.reset_index(drop=True), preds_df], axis=1)
        st.write("Predictions:")
        st.dataframe(out.head(50), use_container_width=True)

        out_buf = BytesIO()
        with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button("Download predictions (Excel)", data=out_buf.getvalue(), file_name="eca_predictions.xlsx")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
