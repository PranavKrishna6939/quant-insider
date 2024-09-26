import pandas as pd
import numpy as np
import os

call_folder = '//home/jjbigdub/gitrepo/quant-insider/Nifty2023/call'
put_folder = '/home/jjbigdub/gitrepo/quant-insider/Nifty2023/put'
spot_data_file = '/home/jjbigdub/gitrepo/quant-insider/Nifty2023/NIFTY_cash.csv'

def combine_files_from_folder(folder_path):
    combined_df = pd.DataFrame()
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            print(f"Parsing file: {file_path}")
            try:
                temp_df = pd.read_csv(file_path, parse_dates=['datetime'])
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                print(f"Skipped empty or malformed file: {file_path}")
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")
    return combined_df

def calculate_time_to_expiry_and_vol(df):
    df['time_to_expiry'] = (df['expiry_date'] - df['datetime']).dt.total_seconds() / (365 * 24 * 60 * 60)
    df['log_return'] = np.log(df['close_spot'] / df['close_spot'].shift(1))
    df['log_return'].dropna(inplace=True)
    df['sigma_20'] = df['log_return'].rolling(window=20).std() * np.sqrt(390 * 252)
    return df

call_df = combine_files_from_folder(call_folder)
put_df = combine_files_from_folder(put_folder)
spot_df = pd.read_csv(spot_data_file, parse_dates=['datetime'])

call_df['expiry_date'] = pd.to_datetime(call_df['expiry_date'], format='%d-%b-%y')
put_df['expiry_date'] = pd.to_datetime(put_df['expiry_date'], format='%d-%b-%y')
call_df['expiry_date'] = call_df['expiry_date'].apply(lambda x: x.replace(hour=15, minute=30))
put_df['expiry_date'] = put_df['expiry_date'].apply(lambda x: x.replace(hour=15, minute=30))
merged_call_df = pd.merge(call_df, spot_df[['datetime', 'close']], on='datetime', suffixes=('_option', '_spot'))
merged_put_df = pd.merge(put_df, spot_df[['datetime', 'close']], on='datetime', suffixes=('_option', '_spot'))

merged_call_df = calculate_time_to_expiry_and_vol(merged_call_df)
merged_put_df = calculate_time_to_expiry_and_vol(merged_put_df)

columns_to_drop = ['stock_code', 'exchange_code', 'product_type', 'count', 'open_interest', 'volume']
merged_call_df.drop(columns=columns_to_drop, inplace=True)
merged_put_df.drop(columns=columns_to_drop, inplace=True)

merged_call_df.to_csv('call_data.csv', index=False)
merged_put_df.to_csv('put_data.csv', index=False)
