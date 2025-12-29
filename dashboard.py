import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
import os
import numpy as np


try:
    from mrf_lib import MacroRandomForest
except ImportError:
    st.error("Error: 'mrf_lib.py' not found. Please ensure it is in the same directory.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="MRF Inflation Forecaster", layout="wide")
st.title("Macroeconomic Random Forest (MRF)")
st.markdown("""
Core Paper Logic: Modeling evolving parameters ($y_t = X_t \\beta_t$) where $\\beta_t$ changes based on economic states.
The orange line below shows the Time-Varying Beta — seeing how the economy changes structure over time.
""")

# 1. LOAD DATA & MODEL
@st.cache_data
def load_data_and_predict():
    start_date = "2000-01-01"
    
    
    tickers = {"Wheat": "ZW=F", "Oil": "CL=F", "Dollar": "DX-Y.NYB"}
    data_frames = []
    
    for name, ticker in tickers.items():
        try:
            df_ticker = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df_ticker.columns, pd.MultiIndex): 
                df_ticker.columns = df_ticker.columns.droplevel(1)
            col = 'Adj Close' if 'Adj Close' in df_ticker.columns else 'Close'
            df_ticker = df_ticker[col].resample('MS').mean().rename(f"{name}_Price")
            data_frames.append(df_ticker)
        except Exception as e:
            pass

    # Load CPI
    if os.path.exists("cpi.csv"): 
        path = "cpi.csv"
    elif os.path.exists("data/cpi.csv"): 
        path = "data/cpi.csv"
    else: 
        return None 

    cpi = pd.read_csv(path, parse_dates=['observation_date'], index_col='observation_date')
    cpi = cpi.loc[start_date:]
    cpi.columns = ["CPI"]
    data_frames.append(cpi)
    
    # Create the main DataFrame 
    df = pd.concat(data_frames, axis=1).dropna()
    df = df.pct_change().dropna()
    
    
    # Linear Regressors (X_t)
    df['Const'] = 1.0
    df['CPI_Lag_1'] = df['CPI'].shift(1)
    df['Wheat_Lag_4'] = df['Wheat_Price'].shift(4)
    df['Oil_Lag_1'] = df['Oil_Price'].shift(1)
    
    # State Variables (S_t)
    df['Wheat_Vol'] = df['Wheat_Price'].rolling(6).std().shift(4)
    df['Wheat_Trend'] = df['Wheat_Price'].rolling(6).mean().shift(4)
    df['Oil_Vol'] = df['Oil_Price'].rolling(3).std().shift(1)
    df['Dollar_Lag'] = df['Dollar_Price'].shift(3)
    df['Month'] = df.index.month
    
    return df.dropna()

# Execute Loader
raw_df = load_data_and_predict()

if raw_df is None:
    st.error("File Error: Could not find 'cpi.csv'. Please ensure it exists.")
    st.stop()

# Load Model Bundle
try:
    bundle = joblib.load("models/mrf_model.pkl")
    model = bundle['model']
    scaler_X = bundle['scaler_X']
    scaler_S = bundle['scaler_S']
    X_cols = bundle['X_cols']
    S_cols = bundle['S_cols']
    
    if 'X_col_names_no_const' in bundle:
        X_names_no_const = bundle['X_col_names_no_const']
    else:
       
        X_names_no_const = [c for c in X_cols if c != 'Const']
except FileNotFoundError:
    st.error("Model Error: 'models/mrf_model.pkl' not found. Please run 'train_model.py' first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


try:
    S_scaled = scaler_S.transform(raw_df[S_cols])

    
    X_vals_no_const = raw_df[X_names_no_const]
    X_scaled_vals = scaler_X.transform(X_vals_no_const)
    
    X_final = np.column_stack([np.ones(len(raw_df)), X_scaled_vals])
except KeyError as e:
    st.error(f"Data Mismatch: Model expects columns {e} which are missing. Re-run train_model.py.")
    st.stop()


betas = model.predict_gtvps(S_scaled)
df_betas = pd.DataFrame(betas, index=raw_df.index, columns=X_cols)


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generalized Time-Varying Parameters (GTVPs)")
    
    
    param_to_view = st.selectbox("Select Parameter to Visualize:", X_cols, index=2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_betas.index, y=df_betas[param_to_view], 
                             mode='lines', name=f'Beta: {param_to_view}', 
                             line=dict(color='#FF4B4B', width=2)))
    
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(title=f"How sensitivity to '{param_to_view}' changes over time", 
                      yaxis_title="Beta Value (Impact Strength)",
                      xaxis_title="Year",
                      height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Current Economic Regime")
    last_row = raw_df.iloc[-1]
    last_beta = df_betas.iloc[-1]
    
    st.markdown("Latest Sensitivities")
    st.metric("Wheat Sensitivity", f"{last_beta['Wheat_Lag_4']:.4f}", help="If Wheat goes up 1%, Inflation goes up X%")
    st.metric("Oil Sensitivity", f"{last_beta['Oil_Lag_1']:.4f}", help="If Oil goes up 1%, Inflation goes up X%")
   
    pred_val = np.sum(X_final[-1] * last_beta.values) 
   
    st.metric("Predicted Inflation Change", f"{pred_val*100:.3f}%", delta_color="inverse")

st.divider()
st.info("Note how the coefficients (Betas) are not flat lines. They react to the State Variables (Volatility, Trends). This is the 'Random Forest' deciding that specific economic rules apply only in specific times.")