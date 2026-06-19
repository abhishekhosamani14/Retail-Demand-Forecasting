import os
import joblib
import pandas as pd
import numpy as np

def load_model_and_scaler(model_dir="models"):
    """
    Load the trained model and scaler.
    """
    model_path = os.path.join(model_dir, "xgboost_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler file not found. Please run training pipeline first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_features_for_prediction(store, item, price, date_str, historical_df):
    """
    Dynamically extract date features and lookup/calculate lag and rolling features
    for a specific store, item, price, and target date.
    """
    target_date = pd.to_datetime(date_str)
    
    # 1. Date Features
    year = target_date.year
    month = target_date.month
    week = int(target_date.isocalendar().week)
    day = target_date.day
    dayofweek = target_date.dayofweek
    quarter = target_date.quarter
    is_weekend = 1 if dayofweek in [5, 6] else 0
    
    # Filter historical data for this store and item
    hist = historical_df[(historical_df['store'] == store) & (historical_df['item'] == item)].copy()
    hist['date'] = pd.to_datetime(hist['date'])
    hist = hist.sort_values('date').reset_index(drop=True)
    
    # 2. Lags and Rolling Means (Historical sales lookup)
    # Filter historical sales before target date
    past_sales = hist[hist['date'] < target_date]
    
    # Defaults in case history is insufficient
    default_sales = hist['sales'].mean() if len(hist) > 0 else 50.0
    
    # Helper to get sales on specific past date
    def get_lag_sales(lag_days):
        lag_date = target_date - pd.Timedelta(days=lag_days)
        match = past_sales[past_sales['date'] == lag_date]
        if not match.empty:
            return float(match.iloc[0]['sales'])
        # Fallback 1: Positional index check if exact date is not found
        # (e.g., if there are gaps in date, get the value lag_days steps back in past_sales)
        if len(past_sales) >= lag_days:
            return float(past_sales.iloc[-lag_days]['sales'])
        return default_sales

    # Helper to get rolling mean for a window prior to target date
    def get_rolling_mean(window):
        # Get sales for the window days immediately preceding target_date
        window_sales = past_sales.tail(window)['sales']
        if len(window_sales) > 0:
            # If we don't have enough rows, pad with default_sales
            if len(window_sales) < window:
                padding = [default_sales] * (window - len(window_sales))
                return float(np.mean(list(window_sales) + padding))
            return float(window_sales.mean())
        return default_sales

    sales_lag_7 = get_lag_sales(7)
    sales_lag_14 = get_lag_sales(14)
    sales_lag_30 = get_lag_sales(30)
    
    sales_roll_mean_7 = get_rolling_mean(7)
    sales_roll_mean_14 = get_rolling_mean(14)
    sales_roll_mean_30 = get_rolling_mean(30)
    
    # Build feature dictionary in correct order
    feature_dict = {
        'store': store,
        'item': item,
        'price': price,
        'year': year,
        'month': month,
        'week': week,
        'day': day,
        'dayofweek': dayofweek,
        'quarter': quarter,
        'is_weekend': is_weekend,
        'sales_lag_7': sales_lag_7,
        'sales_lag_14': sales_lag_14,
        'sales_lag_30': sales_lag_30,
        'sales_roll_mean_7': sales_roll_mean_7,
        'sales_roll_mean_14': sales_roll_mean_14,
        'sales_roll_mean_30': sales_roll_mean_30
    }
    
    return feature_dict

def predict_demand(store, item, price, date_str, historical_df, model=None, scaler=None):
    """
    Perform end-to-end demand prediction.
    """
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
        
    # Get features
    feat_dict = get_features_for_prediction(store, item, price, date_str, historical_df)
    
    # Convert to DataFrame to ensure correct column order
    features_ordered = [
        'store', 'item', 'price',
        'year', 'month', 'week', 'day', 'dayofweek', 'quarter', 'is_weekend',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
        'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30'
    ]
    
    X_df = pd.DataFrame([feat_dict])[features_ordered]
    
    # Scale features
    X_scaled = scaler.transform(X_df)
    
    # Predict
    pred = model.predict(X_scaled)[0]
    
    # Demand must be non-negative
    return max(0.0, float(pred))
