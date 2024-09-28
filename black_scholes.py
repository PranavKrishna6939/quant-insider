import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
import keras
tqdm.pandas()

model = keras.models.load_model('saved/mlp1-135.keras')

call_df = pd.read_csv('/home/jjbigdub/gitrepo/quant-insider/call_data.csv')
call_df = call_df.drop(columns=['datetime', 'expiry_date', 'right','open','high','low' ,'log_return'])
call_df = call_df.dropna(axis=0)
def black_scholes(row):
    S = row.close_spot
    X = row.strike_price 
    T = row.time_to_expiry
    r = 0.06
    σ = row.sigma_20
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    C = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return C

call_df['black_scholes_pred'] = call_df.progress_apply(black_scholes, axis=1)
X_model = call_df[['close_spot', 'strike_price', 'time_to_expiry', 'sigma_20']].values
call_df['model_pred'] = model.predict(X_model, verbose=1)
mse = lambda df, pred_col: np.mean(np.square(df['close_option'] - df[pred_col]))
mse_black_scholes = mse(call_df, 'black_scholes_pred')
mse_model = mse(call_df, 'model_pred')

print(f"Black-Scholes MSE: {mse_black_scholes}")
print(f"Model MSE: {mse_model}")

call_df[['close_option', 'black_scholes_pred', 'model_pred']].to_csv('call_preds.csv', index=False)
