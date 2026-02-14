import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import gc
import matplotlib.pyplot as plt

# ==========================================
# 1. HELPER FUNCTIONS & CONFIG
# ==========================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def add_row_stats(df):
    """
    Performs the row-wise feature engineering from the notebook.
    """
    df = df.copy()
    # Select numerical columns only
    num = df.select_dtypes(include=[np.number])
    
    df["row_mean"] = num.mean(axis=1)
    df["row_std"]  = num.std(axis=1)
    df["row_min"]  = num.min(axis=1)
    df["row_max"]  = num.max(axis=1)
    df["row_sum"]  = num.sum(axis=1)
    return df

# ==========================================
# 2. STREAMLIT UI LAYOUT
# ==========================================

st.set_page_config(page_title="Flood Prediction Model", layout="wide")

st.title("🌊 Predict the Flood: Kaggle Model Dashboard")
st.markdown("""
This app runs the **LightGBM + CatBoost** ensemble pipeline from your Jupyter Notebook.
Upload your training and test data below to generate predictions.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("1. Data Upload")
train_file = st.sidebar.file_uploader("Upload train.csv", type=["csv"])
test_file = st.sidebar.file_uploader("Upload test.csv", type=["csv"])

st.sidebar.header("2. Model Config")
n_folds = st.sidebar.slider("Number of CV Folds", min_value=2, max_value=10, value=5)
enable_catboost = st.sidebar.checkbox("Enable CatBoost", value=True)

st.sidebar.header("3. Blending Weights")
weight_lgb = st.sidebar.slider("LightGBM Weight", 0.0, 1.0, 0.6, 0.05)
weight_cb = 1.0 - weight_lgb
st.sidebar.write(f"**CatBoost Weight:** {weight_cb:.2f}")

# --- Main Execution Block ---

if train_file and test_file:
    # 1. Load Data
    with st.spinner("Loading and preprocessing data..."):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        
        # Identify Target
        TARGET = "FloodProbability"
        if TARGET not in train.columns:
            st.error(f"Target column '{TARGET}' not found in training data.")
            st.stop()
            
        # Separate X and y
        X = train.drop(columns=[TARGET])
        y = train[TARGET].values
        
        # Align Test Columns
        test = test[X.columns]
        
        # Feature Engineering
        X_fe = add_row_stats(X)
        test_fe = add_row_stats(test)
        
        st.success(f"Data Loaded! Train: {X_fe.shape}, Test: {test_fe.shape}")
        
        # Preview Data
        with st.expander("Preview Feature Engineering"):
            st.dataframe(X_fe.head())

    # 2. Training Button
    if st.button("🚀 Train & Predict"):
        
        # Setup Cross-Validation
        bins = pd.qcut(train[TARGET], q=20, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # --- LightGBM Training ---
        st.subheader("1. LightGBM Training")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 256,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 2.0,
            "verbosity": -1,
            "seed": 42,
        }
        
        oof_lgb = np.zeros(len(train))
        pred_test_lgb = np.zeros(len(test_fe))
        fold_scores_lgb = []
        
        # LGBM Loop
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, bins), 1):
            status_text.text(f"Training LightGBM Fold {fold}/{n_folds}...")
            
            X_tr, X_va = X_fe.iloc[tr_idx], X_fe.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            dtr = lgb.Dataset(X_tr, label=y_tr)
            dva = lgb.Dataset(X_va, label=y_va)

            model = lgb.train(
                lgb_params,
                dtr,
                num_boost_round=2000, # Reduced from 20000 for Streamlit performance
                valid_sets=[dva],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                ],
            )

            va_pred = model.predict(X_va, num_iteration=model.best_iteration)
            oof_lgb[va_idx] = va_pred
            
            fold_rmse = rmse(y_va, va_pred)
            fold_scores_lgb.append(fold_rmse)
            pred_test_lgb += model.predict(test_fe, num_iteration=model.best_iteration) / n_folds
            
            progress_bar.progress(int((fold / n_folds) * 100))
            st.write(f"🔹 **LGBM Fold {fold}:** RMSE = {fold_rmse:.6f}")
            
            del model, dtr, dva, X_tr, X_va, y_tr, y_va
            gc.collect()

        st.success(f"LightGBM Finished! Mean RMSE: {np.mean(fold_scores_lgb):.6f}")

        # --- CatBoost Training ---
        pred_test_cb = np.zeros(len(test_fe))
        
        if enable_catboost:
            st.subheader("2. CatBoost Training")
            try:
                from catboost import CatBoostRegressor
                progress_bar_cb = st.progress(0)
                status_text_cb = st.empty()
                fold_scores_cb = []
                
                for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, bins), 1):
                    status_text_cb.text(f"Training CatBoost Fold {fold}/{n_folds}...")
                    
                    X_tr, X_va = X_fe.iloc[tr_idx], X_fe.iloc[va_idx]
                    y_tr, y_va = y[tr_idx], y[va_idx]

                    cb = CatBoostRegressor(
                        loss_function="RMSE",
                        iterations=2000, # Reduced for Streamlit
                        learning_rate=0.03,
                        depth=8,
                        l2_leaf_reg=6,
                        random_seed=42,
                        verbose=0,
                        od_type="Iter",
                        od_wait=100
                    )
                    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

                    va_pred = cb.predict(X_va)
                    fold_rmse = rmse(y_va, va_pred)
                    fold_scores_cb.append(fold_rmse)
                    pred_test_cb += cb.predict(test_fe) / n_folds

                    progress_bar_cb.progress(int((fold / n_folds) * 100))
                    st.write(f"🔸 **CatBoost Fold {fold}:** RMSE = {fold_rmse:.6f}")
                
                st.success(f"CatBoost Finished! Mean RMSE: {np.mean(fold_scores_cb):.6f}")
                
            except ImportError:
                st.warning("CatBoost not installed. Skipping...")
                enable_catboost = False

        # --- Blending & Submission ---
        st.subheader("3. Final Blending & Submission")
        
        if enable_catboost:
            final_pred = (weight_lgb * pred_test_lgb) + (weight_cb * pred_test_cb)
            st.info(f"Applied Blending: {weight_lgb:.2f} LGBM + {weight_cb:.2f} CatBoost")
        else:
            final_pred = pred_test_lgb
            st.info("Using LightGBM predictions only.")

        # Prepare Submission DataFrame
        # Attempt to find an ID column in Test, otherwise use Index
        id_col = next((c for c in test.columns if c.lower() in ["id", "index"]), None)
        
        sub_df = pd.DataFrame()
        if id_col:
            sub_df[id_col] = test[id_col]
        else:
            sub_df["id"] = test.index
            
        sub_df[TARGET] = final_pred
        
        # Display & Download
        st.dataframe(sub_df.head())
        
        csv = sub_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download submission.csv",
            data=csv,
            file_name="submission.csv",
            mime="text/csv",
        )

else:
    st.info("👋 Please upload `train.csv` and `test.csv` in the sidebar to begin.")