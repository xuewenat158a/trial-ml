
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from io import BytesIO

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib

st.set_page_config(page_title="ECA Multi‑Output RF (J & CTOD)", layout="wide")

st.title("ECA Multi‑Output Random Forest: Predict J & CTOD")

with st.expander("About this app", expanded=False):
    st.markdown(
        "- Train a multi‑output Random Forest from 9 inputs to predict **J*** and **CTOD*** columns.\n"
        "- Handles categorical features (`Geometry`, `Notch Location`) via one‑hot.\n"
        "- Shows per‑target metrics and aggregated permutation feature importance.\n"
        "- Try your own single‑row inputs or batch‑predict by uploading a file."
    )

# ---------- Sidebar Controls ----------
st.sidebar.header("Settings")

DEFAULT_FILE = "ECA ML.xlsx"
input_choice = st.sidebar.selectbox(
    "Choose data source",
    ("Use bundled path", "Upload Excel file (.xlsx)")
)

if input_choice == "Use bundled path":
    input_path = st.sidebar.text_input("Excel path", value=DEFAULT_FILE)
    sheet_name = st.sidebar.text_input("Sheet name", value="ML database")
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])
    sheet_name = st.sidebar.text_input("Sheet name", value="ML database")
    input_path = None

test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
n_estimators = st.sidebar.slider("n_estimators", min_value=200, max_value=1000, value=500, step=50)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
run_cv = st.sidebar.checkbox("Run 5‑fold CV (average R²)", value=False)

st.sidebar.markdown("---")
do_save = st.sidebar.checkbox("Save trained model", value=True)
model_filename = st.sidebar.text_input("Model filename", value="eca_multioutput_rf.joblib")

# ---------- Load Data ----------
@st.cache_data(show_spinner=False)
def load_excel(_uploaded_file, _input_path, _sheet):
    if _uploaded_file is not None:
        df = pd.read_excel(_uploaded_file, sheet_name=_sheet)
    else:
        df = pd.read_excel(_input_path, sheet_name=_sheet)
    # normalize column names
    df.columns = [c.strip().replace("\u200b", "") for c in df.columns]
    rename_map = {
        "Sour region": "Sour Region",
        "ppH2S(bara)": "ppH2S",
        "Test Pressure(bara)": "Test Pressure",
        "a/w": "a/W",
    }
    df.rename(columns=rename_map, inplace=True)
    return df

try:
    df = load_excel(uploaded_file, input_path, sheet_name)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.subheader("Preview data")
st.dataframe(df.head(20), use_container_width=True)

# ---------- Feature / Target Selection ----------
all_cols = list(df.columns)

default_features = ['Sour Region','pH','ppH2S','Test Pressure','K-rate','a/W','Geometry','Notch Location','CI']
missing_feats = [c for c in default_features if c not in all_cols]
if missing_feats:
    st.warning(f"These default feature columns are missing: {missing_feats}")
feature_cols = st.multiselect("Feature columns", options=all_cols, default=[c for c in default_features if c in all_cols])

# Targets: pick all columns starting with J or CTOD by default
auto_targets = [c for c in all_cols if c.startswith("J") or c.startswith("CTOD")]
if not auto_targets:
    st.error("No target columns detected (columns starting with 'J' or 'CTOD'). Please select targets manually below.")
target_cols = st.multiselect("Target columns", options=[c for c in all_cols if c not in feature_cols], default=auto_targets)

if len(feature_cols) == 0 or len(target_cols) == 0:
    st.error("Select at least 1 feature and 1 target.")
    st.stop()

# ---------- Build Train/Test ----------
cat_cols = [c for c in ["Geometry","Notch Location"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Basic NA handling: drop rows with NA in features or targets
data = df[feature_cols + target_cols].dropna(axis=0).reset_index(drop=True)
if len(data) < 10:
    st.warning("Very small dataset after dropping NAs; results may be unstable.")
X = data[feature_cols]
y = data[target_cols]

@st.cache_resource(show_spinner=True)
def train_model(X, y, num_cols, cat_cols, test_size, n_estimators, random_state):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", preprocess), ("model", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe.fit(X_train, y_train)

    return pipe, (X_train, X_test, y_train, y_test)

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
    st.subheader("Per‑target metrics")
    st.dataframe(metrics_df, use_container_width=True)
with col2:
    st.metric("Avg R²", f"{metrics_df['R2'].mean():.3f}")
    st.metric("Avg RMSE", f"{metrics_df['RMSE'].mean():.3f}")

# Optional CV
if run_cv:
    st.info("Running 5‑fold CV on full data (this may take a while).")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # For multi-output, sklearn averages scores across targets internally
    scores = cross_validate(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    st.write(f"CV R² (mean ± std): {scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}")

# ---------- Permutation Importance (aggregated) ----------
def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    numeric_names = list(numeric_features)
    if len(categorical_features) > 0:
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(categorical_features))
    else:
        cat_names = []
    return numeric_names + cat_names

with st.spinner("Computing permutation importance..."):
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
expanded_names = get_transformed_feature_names(
    pipe.named_steps["prep"], num_cols, cat_cols
)
pi_df = pd.DataFrame({
    "expanded_feature": expanded_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

def base_feature(name):
    # OneHot names like "Geometry_SENB" -> "Geometry"
    if any(name.startswith(c) for c in cat_cols):
        return name.split("_", 1)[0]
    return name

pi_df["base_feature"] = pi_df["expanded_feature"].apply(base_feature)
agg = (pi_df.groupby("base_feature")[["importance_mean"]]
       .sum()
       .sort_values("importance_mean", ascending=False)
       .reset_index())

st.subheader("Permutation importance (aggregated to original features)")
st.dataframe(agg, use_container_width=True)

# Bar chart (matplotlib, single plot, default colors)
fig, ax = plt.subplots(figsize=(8, max(3, len(agg)*0.4)))
ax.barh(agg["base_feature"], agg["importance_mean"])
ax.invert_yaxis()
ax.set_title("Permutation Importance")
ax.set_xlabel("Mean importance (validation)")
ax.set_ylabel("Feature")
st.pyplot(fig, clear_figure=True)

# ---------- Save model ----------
if do_save:
    try:
        joblib.dump(pipe, model_filename)
        st.success(f"Saved model to: {model_filename}")
        with open(model_filename, "rb") as f:
            st.download_button("Download model file", f, file_name=model_filename)
    except Exception as e:
        st.warning(f"Could not save model: {e}")

st.markdown("---")

# ---------- Inference (single row) ----------
st.subheader("Try a single prediction")
with st.form("single_pred"):
    inputs = {}
    for col in feature_cols:
        if col in cat_cols:
            # categories from training set
            cats = sorted(list(pd.Series(X[col]).dropna().unique()))
            default_val = cats[0] if len(cats) else ""
            inputs[col] = st.selectbox(col, options=cats, index=0 if default_val in cats else 0)
        else:
            # numeric
            series = pd.Series(X[col]).dropna()
            default_val = float(series.median()) if len(series) else 0.0
            inputs[col] = st.number_input(col, value=default_val, step=0.001, format="%.6f")
    submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            row = pd.DataFrame([inputs])
            pred = pipe.predict(row)
            pred_df = pd.DataFrame(pred, columns=target_cols)
            st.write("Prediction:")
            st.dataframe(pred_df, use_container_width=True)
            # download as CSV
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

        # Ensure required columns exist; fill missing with training medians for numeric, first seen cat for categoricals
        for c in feature_cols:
            if c not in newX.columns:
                if c in cat_cols:
                    newX[c] = pd.Series([pd.Series(X[c]).dropna().unique()[0] if X[c].dropna().size else "" ] * len(newX))
                else:
                    val = float(pd.Series(X[c]).dropna().median()) if X[c].dropna().size else 0.0
                    newX[c] = val

        preds = pipe.predict(newX[feature_cols])
        preds_df = pd.DataFrame(preds, columns=target_cols)
        out = pd.concat([newX.reset_index(drop=True), preds_df], axis=1)
        st.write("Predictions:")
        st.dataframe(out.head(50), use_container_width=True)

        # download
        out_buf = BytesIO()
        with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button("Download predictions (Excel)", data=out_buf.getvalue(), file_name="eca_predictions.xlsx")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
