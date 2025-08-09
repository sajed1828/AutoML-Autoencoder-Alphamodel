import pandas as pd
import numpy as np
import time
import os
import requests
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from smartmoneyconcepts.smc import smc


from Price_Action import PriceAction
from modul_features_map import genlenDataset, Autoencoder, train_autoencoder
from SMC_ML import  (
                      add_liquidity,
                      fig_to_buffer, 
                      add_FVG, 
                      add_swing_highs_lows, 
                      add_bos_choch, add_OB, 
                      add_liquidity, 
                      add_previous_high_low, 
                      add_sessions, 
                      add_retracements,
                      )
from alpha_factor import Alpha_Zero, alpha_factor


class genlenDataset(Dataset):
    def __init__(self, df, FEATURE_COLUMNS, qen_len=10):
        self.qen_len = qen_len
        self.scaler = MinMaxScaler()
        
        data = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
        scaled = self.scaler.fit_transform(data)
        self.target = scaled[:, FEATURE_COLUMNS.index("close")]
        self.data = scaled

    def __len__(self):
        return len(self.data) - self.qen_len
    
    def __getitem__(self, idx):
        X = self.data[idx: idx+self.qen_len]
        y = self.target[idx+self.qen_len]
        return torch.tensor(X,dtype= torch.float), torch.tensor(y,dtype= torch.float)     


class LSTM_modul(nn.Module):
    def __init__(self, input_size=14, hiddan_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hiddan_size, 6)
        self.droup = nn.Dropout(0.2)
        self.fc = nn.Linear(hiddan_size , 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.droup(out[:, -1 ,:])
        return self.fc(out)   

def train_model(dataloader, model, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device) , y_batch.to(device)
            optimizer.total_loss = 0
            optimizer.zero_grad()
            output = model(x_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(dataloader)

def predict_next(df, FEATURE_COLUMNS, model, scaler, seq_len=10):
    df = df[FEATURE_COLUMNS].copy()
    df = df.replace([np.inf, - np.inf], np.nan).dropna()
    
    scaled = scaler.transform(df.values[-seq_len:])

    input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    
    with torch.no_grad():
        predictions =  model(input_tensor).cpu().numpy()
    
    
    dummy = np.zeros((1, len(FEATURE_COLUMNS)))
    close_index = FEATURE_COLUMNS.index("close")
        
    dummy[0, 3] = predictions[0][0]
    inv =scaler.inverse_transform(dummy)
    return inv[0, close_index] 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modul = LSTM_modul().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(modul.parameters(), lr=0.01)
        

def get_latest_candles(symbol="BTCUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        data = requests.get(url).json()
    except Exception as e:
        print("❌ Error fetching data:", e)
        return None

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df = df.astype({
        'open': float, 'high': float, 'low': float,
        'close': float, 'volume': float
    })
    return df.dropna()


gif = []

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
    
    df = pd.read_csv("BITCOIN.csv")
    rf = pd.read_csv("BITCOIN_PREDICTED.csv")
    file_exists = os.path.exists("BITCOIN_PREDICTED.csv")

    df['returns'] = df['close'].pct_change()
    df['vwap'] = (((df['high'] + df['low'] + df['close']) / 3) * df['volume']).cumsum() / df['volume'].cumsum()


    if os.path.exists("lstm_model.pth"):
        FEATURE_COLUMNS = rf.select_dtypes(include=("float64", "uint64")).columns.tolist()
        print(f"✅ Features ({len(FEATURE_COLUMNS)}):", FEATURE_COLUMNS)
        dataset = genlenDataset(df=df, FEATURE_COLUMNS=FEATURE_COLUMNS, qen_len=31)
        if len(dataset) == 0:
            print("⚠️ Not enough data for training.")
            exit()
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        sample_x, _ = dataset[0]
        input_dim = sample_x.numel()

        modul = LSTM_modul(input_size=input_dim).to(device)
        modul.load_state_dict(torch.load("lstm_model.pth"))
        modul.eval()

    else:
        if not df.empty:
            last_row = rf.sort_values(by="timestamp").iloc[-1]

            start_date = pd.to_datetime(last_row)
        df = df[df['timestamp'] >= start_date].reset_index(drop=True)
        
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

        df.to_csv("BITCOIN_PREDICTED.csv", mode='a', header=not file_exists, index=False)
        
       
        FEATURE_COLUMNS = df.select_dtypes(include=("float64", "uint64")).columns.tolist()
        #FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if df[col].notna().all()]
        print(f"✅ Features ({len(FEATURE_COLUMNS)}):", FEATURE_COLUMNS)

        dataset = genlenDataset(df=df, FEATURE_COLUMNS=FEATURE_COLUMNS, qen_len=31)
        if len(dataset) == 0:
            print("⚠️ Not enough data for training.")
            exit()

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        sample_x, _ = dataset[0]
        input_dim = sample_x.numel()

        model = Autoencoder(input_dim=input_dim, encoding_dim=10)

        input_size = len(FEATURE_COLUMNS)
        model_lstm = LSTM_modul(input_size=input_size).to(device)
        optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        initial_loss = train_model(dataloader, model_lstm, criterion, optimizer, epochs=3)
        print(f"🧪 Initial training loss: {initial_loss:.4f}")
        torch.save(model_lstm.state_dict(), "lstm_model.pth")

    while True:
        try:
            new_df = get_latest_candles(symbol="BTCUSDT", interval="1m", limit=100)
            if not df.empty:
                print("DATA SHAPE:", df.shape)
        
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    
                    future_ta = executor.submit(run_ta_indicators, new_df.copy())
                    future_alpha = executor.submit(run_alpha_factors, new_df.copy())
                    future_pa = executor.submit(run_price_action, new_df.copy())

                    df_ta = future_ta.result()
                    df_alpha = future_alpha.result()
                    df_pa = future_pa.result()

                new_df.dropna(inplace=True)

                new_df = pd.concat([new_df, df_ta, df_alpha, df_pa], axis=1)
                print(new_df.head())

                dataset = genlenDataset(new_df, FEATURE_COLUMNS=FEATURE_COLUMNS, qen_len=10)
                dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

                loss = train_model(dataloader, modul, criterion, optimizer, epochs=1)
                prediction = predict_next(new_df, FEATURE_COLUMNS, modul, dataset.scaler, seq_len=10)

                trend = "🔼 Up" if prediction > new_df["close"].iloc[-1] else "🔽 Down"
                text_trend = f"[{datetime.now().strftime('%H:%M:%S')}] 🔮 Close ≈ {prediction:.2f} USDT | Loss: {loss:.4f} | Trend: {trend}"
                print(text_trend)

                torch.save(modul.state_dict(), "model.pth")

                window = 100
                gif.clear()
                for pos in range(window, len(df)):
                    window_df = df.iloc[pos - window:pos]

                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=window_df.index,
                            open=window_df["open"],
                            high=window_df["high"],
                            low=window_df["low"],
                            close=window_df["close"],
                            increasing_line_color="#77dd76",
                            decreasing_line_color="#ff6962",
                        )
                    ])

                    fvg_data = smc.fvg(window_df, join_consecutive=True)
                    swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=5)
                    bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
                    ob_data = smc.ob(window_df, swing_highs_lows_data)
                    liquidity_data = smc.liquidity(window_df, swing_highs_lows_data)
                    previous_high_low_data = smc.previous_high_low(window_df, time_frame="4h")
                    sessions = smc.sessions(window_df, session="London")
                    retracements = smc.retracements(window_df, swing_highs_lows_data)

                    fig = add_FVG(fig, window_df, fvg_data)
                    fig = add_swing_highs_lows(fig, window_df, swing_highs_lows_data)
                    fig = add_bos_choch(fig, window_df, bos_choch_data)
                    fig = add_OB(fig, window_df, ob_data)
                    fig = add_liquidity(fig, window_df, liquidity_data)
                    fig = add_previous_high_low(fig, window_df, previous_high_low_data)
                    fig = add_sessions(fig, window_df, sessions)
                    fig = add_retracements(fig, window_df, retracements)

                    fig.update_layout(
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                        margin=dict(l=0, r=0, b=0, t=0),
                        width=500, height=300,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(12, 14, 18, 1)",
                        font=dict(color="white")
                    )

                    fig.add_annotation(
                        text=text_trend,
                        xref="paper",
                        yref="paper",
                        x=0,
                        y=1.05,
                        showarrow=False,
                        font=dict(size=12, color="white"),
                        align="left",
                        bgcolor="rgba(0, 0, 0, 0.5)",
                        bordercolor="white",
                        borderwidth=1,
                        borderpad=5,
                        opacity=0.8,
                    )


                    gif.append(fig_to_buffer(fig))

            time.sleep(60)

        except Exception as e:

            print("❌ Error during live update:", str(e))
            time.sleep(60)

