import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import itertools
from sklearn.preprocessing import StandardScaler
from mrf_lib import MacroRandomForest

def train():
    print("--- Training Macroeconomic Random Forest (MRF) - Advanced ---")
    start_date = "2000-01-01"
    
   
    tickers = {"Wheat": "ZW=F", "Oil": "CL=F", "Dollar": "DX-Y.NYB"}
    data_frames = []
    
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df = df[col].resample('MS').mean().rename(f"{name}_Price")
            data_frames.append(df)
        except: pass

    if os.path.exists("cpi.csv"): path = "cpi.csv"
    elif os.path.exists("data/cpi.csv"): path = "data/cpi.csv"
    else: raise FileNotFoundError("cpi.csv not found")

    cpi = pd.read_csv(path, parse_dates=['observation_date'], index_col='observation_date')
    cpi = cpi.loc[start_date:]
    cpi.columns = ["CPI"]
    data_frames.append(cpi)
    
    df = pd.concat(data_frames, axis=1).dropna()
    df_pct = df.pct_change().dropna()
    
   
    # A. Linear Regressors (X_t)
    df_pct['Const'] = 1.0 
    df_pct['CPI_Lag_1'] = df_pct['CPI'].shift(1)  
    df_pct['Wheat_Lag_4'] = df_pct['Wheat_Price'].shift(4)
    df_pct['Oil_Lag_1'] = df_pct['Oil_Price'].shift(1)
    
    # B. State Variables (S_t)
    df_pct['Wheat_Vol'] = df_pct['Wheat_Price'].rolling(6).std().shift(4)
    df_pct['Wheat_Trend'] = df_pct['Wheat_Price'].rolling(6).mean().shift(4)
    df_pct['Oil_Vol'] = df_pct['Oil_Price'].rolling(3).std().shift(1)
    df_pct['Dollar_Lag'] = df_pct['Dollar_Price'].shift(3)
    df_pct['Month'] = df_pct.index.month

    df_final = df_pct.dropna()
    
    target = 'CPI'
    
    
    X_cols = ['Const', 'CPI_Lag_1', 'Wheat_Lag_4', 'Oil_Lag_1'] 
    S_cols = ['Wheat_Vol', 'Wheat_Trend', 'Oil_Vol', 'Dollar_Lag', 'Month']
    
    
    scaler_S = StandardScaler()
    S_scaled = scaler_S.fit_transform(df_final[S_cols])
    
    scaler_X = StandardScaler()
    # Separate const before scaling
    X_no_const = df_final[[c for c in X_cols if c != 'Const']]
    X_scaled_vals = scaler_X.fit_transform(X_no_const)
    
    # Reassemble X with the constant 
    X_final = np.column_stack([np.ones(len(df_final)), X_scaled_vals])
    y_final = df_final[target].values
    
    # Split
    train_size = int(len(df_final) * 0.85)
    X_train, X_test = X_final[:train_size], X_final[train_size:]
    S_train, S_test = S_scaled[:train_size], S_scaled[train_size:]
    y_train, y_test = y_final[:train_size], y_final[train_size:]

    print(f"Training Data: {len(X_train)} months | Test Data: {len(X_test)} months")

    # 4. HYPERPARAMETER TUNING LOOP
    param_grid = {
        'ridge_lambda': [0.1, 0.5, 1.0, 2.0],  # Penalty strength
        'podium_zeta': [0.1, 0.5, 0.9]         # Time smoothness (0.9 = very smooth)
    }
    
    best_r2 = -float('inf')
    best_params = {}
    best_model = None
    
    print("\n--- Tuning Hyperparameters ---")
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        
        # Train
        mrf = MacroRandomForest(n_estimators=30, max_depth=4, 
                                ridge_lambda=params['ridge_lambda'], 
                                podium_zeta=params['podium_zeta'])
        mrf.fit(X_train, S_train, y_train)
        
        # Evaluate
        preds = mrf.predict(X_test, S_test)
        # R2 Calculation
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"Params: {params} -> R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = mrf
            best_params = params

    print(f"\n🏆 Best R2: {best_r2:.4f} with {best_params}")
    
    # 5. SAVE BUNDLE (Including Scalers!)
    model_bundle = {
        "model": best_model,
        "X_cols": X_cols,
        "S_cols": S_cols,
        "scaler_X": scaler_X,
        "scaler_S": scaler_S,
        "X_col_names_no_const": [c for c in X_cols if c != 'Const']
    }
    
    joblib.dump(model_bundle, "models/mrf_model.pkl")
    print("Best model saved to models/mrf_model.pkl")

if __name__ == "__main__":
    train()