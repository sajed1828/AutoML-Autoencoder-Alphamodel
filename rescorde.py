from datetime import datetime
import os
import time
from alpha_factor import alpha_factor , Alpha_Zero
from Price_Action import PriceAction 
from modul_features_map import  run_autoencoder
import torch
import pandas as pd 
import numpy as np 



def convert_signals_to_df(signals):
    if not signals:
        return pd.DataFrame(columns=["type", "price", "index"])
    
    df = pd.DataFrame(signals)
    df["index"] = df["index"].astype(int)
    return df.sort_values(by="index").reset_index(drop=True)

def run_ta_indicators(df):
    return alpha_factor.ta_factor_indcators(df)

def run_alpha_factors(df):
    return Alpha_Zero.colpute_all_alpha_factors(df)

def run_price_action(df):
    pa = PriceAction()
    P1, P2, P3, P4 = pa.compute_PA(df[["open", "high", "low", "close", "volume"]].copy())
    return pd.concat([
        convert_signals_to_df(P1),
        convert_signals_to_df(P2),
        convert_signals_to_df(P3),
        convert_signals_to_df(P4)
    ], axis=1)


if __name__ == "__main__":
    from update_dataset_of_bitcoin import UpdateBitcoinData
    import concurrent.futures
    
    main = UpdateBitcoinData()
    main.run()

    df = pd.read_csv("dataset/BITCOIN.csv")
    df['date'] = pd.to_datetime(df['timestamp'])
    file_exists = os.path.exists("dataset/BITCOIN_PREDICTED.csv")

    df['returns'] = df['close'].pct_change()
    df['vwap'] = (((df['high'] + df['low'] + df['close']) / 3) * df['volume']).cumsum() / df['volume'].cumsum()


    if not df.empty:
        print("DATA SHAPE:", df.shape)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            future_ta = executor.submit(run_ta_indicators, df.copy())
            future_alpha = executor.submit(run_alpha_factors, df.copy())
            future_pa = executor.submit(run_price_action, df.copy())

            df_ta = future_ta.result()
            df_alpha = future_alpha.result()
            df_pa = future_pa.result()

        df = pd.concat([df_ta, df_alpha, df_pa], axis=1)
        print(df.head())        
        print(df.shape)

        df.to_csv("dataset/BITCOIN_PREDICTED.csv", mode='a', header=not file_exists, index=False)

        df_encoded = []


             
        encoded_df = run_autoencoder(df, qen_len=72, num_epochs=10, top_n_features=20)

        print(encoded_df.head())
        encoded_df.to_csv("dataset/BITCOIN_encoded_data.csv", mode='a', header=not file_exists, index=False)
