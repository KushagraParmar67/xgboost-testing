import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from datetime import datetime, timedelta
from prediction_versions import update_predictions_csv
import ta


Stocks = ['AAPL', 'RELIANCE.NS', 'WIPRO.NS', 'MRF.NS', 'SBIN.NS']

end_date = datetime.today() - timedelta(days=1)
start_date = end_date.replace(year=end_date.year - 5)
predicts = {}
num_lags = 7

# for fetching historical data of given stock
def get_stock_data(ticker, st_date, ed_date):
    try:
        if ticker == 'AAPL':
            ed_date = datetime.today() - timedelta(days=4)
            print(ed_date)
        df = yf.Ticker(ticker).history(start=st_date, end=ed_date, actions=False)
        if df.empty:
            raise ValueError
        return df
    except ValueError:
        print(f"Data Not Found for stock {ticker}")
        return None

def calculate_indicators(df):
    # Calculate lag features
    for i in range(1, num_lags + 1):
        df[f'Close_t-{i}'] = df['Close'].shift(i)
        df[f'Volume_t-{i}'] = df['Volume'].shift(i)  # adding volume lags
    
    # Volume moving averages
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA10'] = df['Volume'].rolling(10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Technical indicators
    df['MA5'] = df['Close'].rolling(5).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Remove inf and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def training_model(df):
    # Train on historical data
    df = calculate_indicators(df)
    features = (
        [f'Close_t-{i}' for i in range(1, num_lags + 1)] + 
        [f'Volume_t-{i}' for i in range(1, num_lags + 1)] + 
        ['MA5', 'RSI', 'MACD', 'MACD_signal', 'Volume_MA5', 'Volume_MA10', 'Volume_Change']
    )

    # Remove inf and NaN values before training
    X, y = df[features], df['Close']
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]


    # --- TimeSeriesSplit Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=5)
    fold_mse = []
    print("\nTimeSeriesSplit Cross-Validation Results:")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        model_cv = xgb.XGBRegressor()
        model_cv.fit(X_train, y_train)
        preds = model_cv.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        fold_mse.append(mse)
        print(f"  Fold {fold+1}: MSE = {mse:.4f}")
    print(f"Average CV MSE: {np.mean(fold_mse):.4f}\n")

  # Adding the Randomized Seach CV here and declaring the parameters
    param_dist = {
    'n_estimators': np.arange(50, 301, 50),         # Number of trees
    'max_depth': np.arange(2, 8),                   # Tree depth
    'learning_rate': np.linspace(0.01, 0.3, 10),    # Step size shrinkage
    'subsample': np.linspace(0.7, 1.0, 4),          # Row sampling
    'colsample_bytree': np.linspace(0.7, 1.0, 4),   # Feature sampling
    'gamma': np.linspace(0, 0.5, 6),                # Minimum loss reduction
    'reg_alpha': np.linspace(0, 0.2, 5),            # L1 regularization
    'reg_lambda': np.linspace(0.5, 2.0, 4)          # L2 regularization
    }
    # --- Train final model on all data for future prediction ---

    model = xgb.XGBRegressor()
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_mean_squared_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X, y)
    print('Best Parameter found by RandomizedSearchCV:', random_search.best_params_)

    # training with best model params
    best_model = random_search.best_estimator_
    best_model.fit(X, y)
    
    # Prepare prediction dataset
    future_preds = []
    temp_df = df[['Close', 'Volume']].copy()  # Start with historical closes
    
    for _ in range(5):
        # Calculate indicators on current data (historical + predicted)
        temp_with_indicators = calculate_indicators(temp_df.copy())
        
        # Remove inf and NaN values
        temp_with_indicators = temp_with_indicators.replace([np.inf, -np.inf], np.nan)
        temp_with_indicators = temp_with_indicators.dropna()
        
        # Get latest valid row (after indicator calculations)
        if not temp_with_indicators.empty:
            latest_row = temp_with_indicators.iloc[-1][features].values.reshape(1, -1)
            pred = best_model.predict(latest_row)[0]
            future_preds.append(pred)
            
            # Append new prediction to temp_df with next business day
            next_date = temp_df.index[-1] + pd.offsets.BDay(1)
            last_volume = temp_df['Volume'].iloc[-1]
            temp_df.loc[next_date] = [pred, last_volume]
        else:
            break  # Stop if no valid data
    
    return future_preds

if __name__ == '__main__':
    for ticker in Stocks:
        df = get_stock_data(ticker, st_date=start_date, ed_date=end_date)
        if df is None:
            print(f"Skipping {ticker} (no data)")
            continue
            
        predictions = training_model(df)
        predicts[ticker] = [{
            (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d"): float(pred)
        } for i, pred in enumerate(predictions)]

    rows = []
    for stock, preds_list in predicts.items():
        for date_pred in preds_list:
            for date, pred in date_pred.items():
                rows.append({'Date': date, 'Stock': stock, 'Prediction': pred})

    df = pd.DataFrame(rows)
    print(df)
    '''  this is for predicting and saving the result in a csv format 
         If do not want to create the csv for versioning and comparing comment
         the line `update_predictions_csv()` 
    '''
    update_predictions_csv(df, version_name='pred_RS-MA-Vol-Tss-Mse-RSCV-FT')
