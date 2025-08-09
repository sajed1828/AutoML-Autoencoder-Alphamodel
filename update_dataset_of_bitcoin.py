import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os


class UpdateBitcoinData:
    def __init__(self, symbol='BTC/USDT', path="dataset/BITCOIN.csv", limit=1000):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.symbol = symbol
        self.limit = limit
        self.path = path


    def run(self):
        all_data = []

        if os.path.exists(self.path):
            existing_df = pd.read_csv(self.path, parse_dates=['timestamp'])
            if not existing_df.empty:
                last_timestamp = existing_df['timestamp'].max()
                since = int(last_timestamp.timestamp() * 1000) + 60_000
            else:
                since = self.exchange.parse8601('2025-08-03T13:02:00')
        else:
            since = self.exchange.parse8601('2025-08-03T13:02:00')

        file_exists = os.path.exists(self.path)

        while since < self.exchange.milliseconds():
            try:
                print("Fetching data from:", datetime.utcfromtimestamp(since / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', since=since, limit=self.limit)
            except Exception as e:
                print("Error while fetching:", e)
                time.sleep(5)
                continue

            if not ohlcv:
                print("No more data returned.")
                break

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            df.to_csv(self.path, mode='a', header=not file_exists, index=False)
            file_exists = True

            since = ohlcv[-1][0] + 60_000
            time.sleep(0.5)

        print("Download finished.")
        