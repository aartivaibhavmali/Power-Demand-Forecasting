"""
PowerCast — Power Supply/Demand Forecasting Dashboard
=========================================================
Run with: streamlit run app.py

Two modes (selectable in sidebar):
  1. Upload & Train  — upload PJMW_MW_Hourly.xlsx/.csv, builds features,
     trains XGBoost/RandomForest/LightGBM live, shows results.
  2. Load Saved Model — load a pre-trained .pkl model + a pre-saved
     test predictions CSV (from your notebook) for instant results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io
import pickle

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="PowerCast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# THEME — Neutral / Slate-Blue
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f8;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0px;
    }
    .sidebar-title span { color: #38bdf8 !important; }
    .sidebar-tagline {
        font-size: 12px;
        color: #94a3b8 !important;
        margin-bottom: 18px;
    }
    h1, h2, h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    .subtitle-text {
        color: #64748b !important;
        font-size: 14px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 11px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    .chart-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .model-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background-color: #dbeafe;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    'Hour',
    'DayOfWeek',
    'Month',
    'Year',
    'DayOfYear',
    'weekofyear',
    'Is_Holiday',
    'lag_24',
    'lag_168',
    'rolling_mean_24',
    'rolling_std_24'
]
TARGET_COL = 'PJMW_MW'


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
@st.cache_data
def build_features(df_raw):
    df = df_raw.copy()

    # Identify datetime + target columns flexibly
    dt_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            dt_col = c
            break
    if dt_col is None:
        dt_col = df.columns[0]

    target_col = None
    for c in df.columns:
        if c != dt_col and pd.api.types.is_numeric_dtype(df[c]):
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[1]

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)
    df = df.rename(columns={target_col: 'PJMW_MW'})
    df = df[['PJMW_MW']]

    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['DayOfYear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    try:
        import holidays
        us_holidays = holidays.US()
        df['Is_Holiday'] = df.index.to_series().dt.date.map(
            lambda x: 1 if x in us_holidays else 0
        ).values
    except ImportError:
        df['Is_Holiday'] = 0

    df['lag_24'] = df['PJMW_MW'].shift(24)
    df['lag_168'] = df['PJMW_MW'].shift(168)
    df['rolling_mean_24'] = df['PJMW_MW'].rolling(24).mean()
    df['rolling_std_24'] = (
    df['PJMW_MW']
    .rolling(24)
    .std()
    )

    data = df.dropna()
    return data



# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------
@st.cache_resource
def train_models(data, split_date):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = data[data.index < split_date].copy()
    test = data[data.index >= split_date].copy()

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    results = {}

    def evaluate(name, y_true, y_pred):
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2': r2}

    # Baseline
    evaluate('Baseline', y_test, X_test['lag_168'])

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        early_stopping_rounds=50, eval_metric='rmse', random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    test['XGB_Prediction'] = xgb_pred
    evaluate('XGBoost', y_test, xgb_pred)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    test['RF_Prediction'] = rf_pred
    evaluate('Random Forest', y_test, rf_pred)

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                   callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    lgb_pred = lgb_model.predict(X_test)
    test['LGB_Prediction'] = lgb_pred
    evaluate('LightGBM', y_test, lgb_pred)

    #SarimaModel
    train_small = train.tail(1000)

    sarima_model = SARIMAX(
      train_small['PJMW_MW'],
      order=(1,1,1),
      seasonal_order=(1,1,1,24),
      enforce_stationarity=False,
      enforce_invertibility=False
     )

    sarima_fit = sarima_model.fit(disp=False)

    sarima_pred = sarima_fit.forecast(
    steps=len(y_test)
    )

    test['Sarima_Prediction'] = sarima_pred.values

    evaluate(
    'SARIMA',
    y_test,
    sarima_pred
    )
    
    models = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'LightGBM': lgb_model, 'SARIMA':sarima_fit}
    pred_cols = {'XGBoost': 'XGB_Prediction', 'Random Forest': 'RF_Prediction', 'LightGBM': 'LGB_Prediction', 'SARIMA':'Sarima_Prediction'}

    return results, test, y_test, models, pred_cols


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Power<span>Cast</span></div>'
        '<div class="sidebar-tagline">Power demand forecasting dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    mode = st.radio(
        "Data source",
        ["Upload & Train", "Load Saved Results"],
        help="Upload raw data to train models live, or load pre-saved model/predictions."
    )

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        '<p style="font-size:12px; color:#94a3b8 !important;">'
        'PJM West hourly power demand forecasting. Models: XGBoost, '
        'Random Forest, LightGBM, compared against a naive baseline.'
        '</p>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.markdown("# Power demand forecast")
st.markdown('<p class="subtitle-text">Actual vs predicted hourly power demand (PJMW_MW)</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================
# MODE 1 — UPLOAD & TRAIN
# ===========================================================
models = None
if mode == "Upload & Train":
    uploaded_file = st.file_uploader(
        "Upload hourly power demand file (.xlsx or .csv) — needs a datetime column and a numeric demand column",
        type=["xlsx", "csv"]
    )

    if uploaded_file is None:
        st.info("Upload your PJMW_MW_Hourly.xlsx (or similar) file to build features and train models.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    with st.spinner("Building features..."):
        data = build_features(df_raw)

    st.success(f"Loaded {len(data):,} rows after feature engineering "
               f"(from {data.index.min().date()} to {data.index.max().date()})")

    years = sorted(data.index.year.unique())
    default_split_year = years[-1] if len(years) > 1 else years[0]
    split_year = st.selectbox(
        "Test set starts from year",
        years,
        index=len(years) - 1
    )
    split_date = f"{split_year}-01-01"

    with st.spinner("Training models (XGBoost, Random Forest, LightGBM)... this may take a moment"):
        results, test, y_test, models, pred_cols = train_models(data, split_date)

    results_df = pd.DataFrame(results).T.sort_values('RMSE')
    best_model_name = results_df.index[results_df.index != 'Baseline'][0] \
        if len(results_df.index[results_df.index != 'Baseline']) else results_df.index[0]
    best_pred_col = pred_cols.get(best_model_name)


# ===========================================================
# MODE 2 — LOAD SAVED RESULTS
# ===========================================================
else:
    st.markdown(
    '<span style="color:red; font-size:16px;">Upload a CSV of saved test predictions (with    columns: a datetime/index, Actual, and one or more prediction columns e.g. XGB_Prediction).</span>',
    unsafe_allow_html=True
    )

    pred_file = st.file_uploader("Upload predictions CSV", type=["csv"])

    if pred_file is None:
        st.info("Export  test[['PJMW_MW','XGB_Prediction','RF_Prediction','LGB_Prediction','Sarima_Prediction']]` "
        "from your notebook as CSV and upload it here.")
        st.markdown("""
```python
test[['PJMW_MW',
      'XGB_Prediction',
      'RF_Prediction',
      'LGB_Prediction',
      'Sarima_Prediction']].to_csv('test_predictions.csv')
```
""")
        st.stop()

    try:
        test = pd.read_csv(pred_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # Identify datetime column if present
    dt_col = None
    for c in test.columns:
        if 'date' in c.lower() or 'time' in c.lower() or c.lower() == 'unnamed: 0':
            dt_col = c
            break
    if dt_col:
        test[dt_col] = pd.to_datetime(test[dt_col], errors='coerce')
        test = test.set_index(dt_col)

    if 'PJMW_MW' not in test.columns:
        # try common alternatives
        for c in test.columns:
            if 'actual' in c.lower():
                test = test.rename(columns={c: 'PJMW_MW'})
                break

    if 'PJMW_MW' not in test.columns:
        st.error("Could not find an actual values column (expected `PJMW_MW` or `Actual`).")
        st.stop()

    y_test = test['PJMW_MW']

    EXCLUDE_COLS = {
    'Hour',
    'DayOfWeek',
    'Month',
    'Year',
    'DayOfYear',
    'weekofyear',
    'Is_Holiday',
    'lag_24',
    'lag_168',
    'rolling_mean_24',
    'rolling_std_24'
     }

    pred_cols = {}
    for c in test.columns:
        if c != 'PJMW_MW' and c not in EXCLUDE_COLS:
            label = c.replace('_Prediction', '').replace('_', ' ')
            pred_cols[label] = c

    if not pred_cols:
        st.error("No prediction columns found.")
        st.stop()

    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    results = {}

    # Baseline using lag_168 if present
    if 'lag_168' in test.columns:
       rmse = root_mean_squared_error(y_test, test['lag_168'])
       mae = mean_absolute_error(y_test, test['lag_168'])
       mape = np.mean(np.abs((y_test - test['lag_168'])/np.where(y_test == 0, 1, y_test))        ) * 100
       r2 = r2_score(y_test, test['lag_168'])
       results['Baseline'] = {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2': r2}

    for label, col in pred_cols.items():
        pred = test[col]
        rmse = root_mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / np.where(y_test == 0, 1, y_test))) * 100
        r2 = r2_score(y_test, pred)
        results[label] = {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2': r2}

    results_df = pd.DataFrame(results).T.sort_values('RMSE')
    non_baseline = results_df.index[results_df.index != 'Baseline']

    if len(non_baseline) > 0:
      best_model_name = non_baseline[0]
    else:
      best_model_name = results_df.index[0]
      best_pred_col = pred_cols.get(best_model_name)
      models = None


# ===========================================================
# SHARED DISPLAY — KPI ROW
# ===========================================================
best_row = results_df.loc[best_model_name]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best model", best_model_name)
k2.metric("RMSE (MW)", f"{best_row['RMSE']:.1f}")
k3.metric("MAPE", f"{best_row['MAPE (%)']:.2f}%")
k4.metric("R2 score", f"{best_row['R2']:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================
# MODEL COMPARISON TABLE
# ===========================================================
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown("### Model comparison")
st.markdown('<p class="subtitle-text">Sorted by RMSE — lower is better</p>', unsafe_allow_html=True)
st.dataframe(results_df.style.format({
    'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MAPE (%)': '{:.2f}', 'R2': '{:.3f}'
}), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================
# ACTUAL VS PREDICTED — TIME SERIES
# ===========================================================

st.markdown("""
<style>

/* Radio option text */
div[data-testid="stRadio"] label p {
    color: black !important;
    font-weight: 600 !important;
}

/* Radio title ("View") */
div[data-testid="stRadio"] > label {
    color: black !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("### Actual vs predicted demand")

view_window = st.radio(
    "View",
    ["First week of test set", "Full test set"],
    horizontal=True
)

n_points = 168 if view_window == "First week of test set" else len(y_test)

x_vals = list(range(n_points))
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=x_vals, y=y_test.iloc[:n_points].values,
    mode='lines', name='Actual',
    line=dict(color='#1e293b', width=2)
))

color_map = {'XGBoost': '#38bdf8', 'XGB': '#38bdf8',
              'Random Forest': '#f59e0b', 'RF': '#f59e0b',
              'LightGBM': '#10b981', 'LGB': '#10b981'}

for label, col in pred_cols.items():
    if col in test.columns:
        fig_line.add_trace(go.Scatter(
            x=x_vals, y=test[col].iloc[:n_points].values,
            mode='lines', name=label,
            line=dict(width=1.5, dash='dot', color=color_map.get(label, '#a855f7'))
        ))

fig_line.update_layout(
    template="plotly_white",

    plot_bgcolor="white",
    paper_bgcolor="white",

    font=dict(
        color="black",
        size=14
    ),

    xaxis=dict(
        title="Hour (test set index)",
        title_font=dict(color="black"),
        tickfont=dict(color="black")
    ),

    yaxis=dict(
        title="Power demand (MW)",
        title_font=dict(color="black"),
        tickfont=dict(color="black")
    ),

    legend=dict(
        orientation="h",
        y=1.12,
        font=dict(
            color="black",
            size=13
        )
    ),

    margin=dict(l=10, r=10, t=10, b=10),
    height=400
)
st.plotly_chart(fig_line, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# FUTURE FORECAST
# ===========================================================


st.markdown("## 30-Day Future Forecast")

forecast_days = st.slider(
    "Select Forecast Days",
    1,
    30,
    30
)

if models is not None:

    future_dates = pd.date_range(
        start=data.index.max() + pd.Timedelta(hours=1),
        periods=forecast_days * 24,
        freq='h'
    )

    future_df = pd.DataFrame(index=future_dates)

    future_df['Hour'] = future_df.index.hour
    future_df['DayOfWeek'] = future_df.index.dayofweek
    future_df['Month'] = future_df.index.month
    future_df['Year'] = future_df.index.year
    future_df['DayOfYear'] = future_df.index.dayofyear
    future_df['weekofyear'] = future_df.index.isocalendar().week.astype(int)

    future_df['Is_Holiday'] = 0

    future_df['lag_24'] = data['lag_24'].iloc[-1]
    future_df['lag_168'] = data['lag_168'].iloc[-1]

    future_df['rolling_mean_24'] = data['rolling_mean_24'].iloc[-1]
    future_df['rolling_std_24'] = data['rolling_std_24'].iloc[-1]

    best_model = models[best_model_name]

    if best_model_name == 'SARIMA':

        future_df['Forecast_MW'] = best_model.forecast(
            steps=len(future_df)
        ).values

    else:

        future_df['Forecast_MW'] = best_model.predict(
            future_df[FEATURE_COLS]
        )

    fig_future = go.Figure()

    fig_future.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df['Forecast_MW'],
            mode='lines',
            name='Forecast'
        )
    )

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )
# ===========================================================
# FEATURE IMPORTANCE (only if models trained live)
# ===========================================================
if models is not None:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("### Feature importance")
    st.markdown('<p class="subtitle-text">Which features drive each model\'s predictions</p>', unsafe_allow_html=True)

    importance_model = st.selectbox("Model", ['XGBoost', 'Random Forest', 'LightGBM'])
    importances = pd.Series(
        models[importance_model].feature_importances_,
        index=FEATURE_COLS
    ).sort_values()

    fig_imp = go.Figure(data=[go.Bar(
        x=importances.values, y=importances.index,
        orientation='h', marker_color='#38bdf8'
    )])
    fig_imp.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="black",
        xaxis_title="Importance",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# DOWNLOAD RESULTS
# ===========================================================
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown("### Export")
csv_buffer = io.StringIO()
results_df.to_csv(csv_buffer)
st.download_button(
    "Download model comparison (CSV)",
    data=csv_buffer.getvalue(),
    file_name="model_comparison.csv",
    mime="text/csv"
)
st.markdown('</div>', unsafe_allow_html=True)
