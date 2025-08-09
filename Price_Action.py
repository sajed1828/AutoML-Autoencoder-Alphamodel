import numpy as np
import pandas as pd


class PriceAction:
    def __init__(self):
        pass

    def __detect_swings(self, df):
        df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        return df

    def __detect_choch(self, df):
        df = self.__detect_swings(df)
        choch_signals = []
        trend = None
        last_swing_high = None
        last_swing_low = None

        for i in range(2, len(df)):
            row = df.iloc[i]

            if row['swing_high']:
                last_swing_high = row['high']

            if row['swing_low']:
                last_swing_low = row['low']

            if last_swing_high and df.iloc[i - 1]['high'] < last_swing_high:
                new_trend = 'up'
                if trend != new_trend:
                    choch_signals.append({'type': 'up', 'price': last_swing_high, 'index': i})
                    trend = new_trend

            elif last_swing_low and df.iloc[i - 1]['low'] > last_swing_low:
                new_trend = 'down'
                if trend != new_trend:
                    choch_signals.append({'type': 'down', 'price': last_swing_low, 'index': i})
                    trend = new_trend

        return choch_signals

    def __detect_fvg(self, df):
        fvg_zones = []
        for i in range(2, len(df)):
            prev_low = df['low'].iloc[i - 2]
            mid_high = df['high'].iloc[i - 1]
            curr_low = df['low'].iloc[i]

            if mid_high < curr_low:
                fvg_zones.append({'index': i, 'low': prev_low, 'mid_high': mid_high, 'high': curr_low})
        return fvg_zones

    def __detect_ob(self, df):

        ob_zones = []
        for i in range(1, len(df)):
            change = abs((df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i])
            if change > 0.02:
                ob_zones.append({'index': i, 'open': df['open'].iloc[i], 'close': df['close'].iloc[i]})
        return ob_zones

    def __detect_bos(self, df):
        bos_points = []
        highs = df['high']
        lows = df['low']
        for i in range(2, len(df)):
            if highs[i] > highs[i - 2]:
                bos_points.append({'index': i, 'type': 'bullish', 'price': highs[i]})
            elif lows[i] < lows[i - 2]:
                bos_points.append({'index': i, 'type': 'bearish', 'price': lows[i]})
        return bos_points

    def compute_PA(self, df):
        P1 = self.__detect_choch(df)
        P2 = self.__detect_ob(df)
        P3 = self.__detect_fvg(df)
        P4 = self.__detect_bos(df)

        return P1, P2, P3, P4 

def convert_signals_to_df(signals):
    if not signals:
        return pd.DataFrame(columns=["type", "price", "index"])
    
    df = pd.DataFrame(signals)
    df["index"] = df["index"].astype(int)
    return df.sort_values(by="index").reset_index(drop=True)
 

if __name__ == "__main__":
    df = pd.read_csv("BITCOIN.csv")

    dataset = df.head(500)
    print("##DATASET##:",
          dataset)

    pa = PriceAction()
    P1, P2, P3, P4 = pa.compute_PA(df)
    new_df = pd.concat([
                    df,
                    convert_signals_to_df(P1),
                    convert_signals_to_df(P2),
                    convert_signals_to_df(P3),
                    convert_signals_to_df(P4)
                ], axis=1)
    print("##COMPUTE_PA##:", new_df)
    new_df.to_csv('BITCOIN_PA_test.csv', index=False)

    