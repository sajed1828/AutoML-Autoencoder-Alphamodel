import pandas as pd
import numpy as np
import ta

class alpha_factor:
    def __init__(self):
        pass

    @staticmethod
    def validate_data(df):
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df

    @staticmethod
    def preprocess_data(df):
        df = alpha_factor.validate_data(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    @staticmethod
    def add_momentum_indicators(df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
        df['wr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        return df

    @staticmethod
    def add_trend_indicators(df):
        macd = ta.trend.MACD(df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        return df

    @staticmethod
    def add_volatility_indicators(df):
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()

        return df

    @staticmethod
    def add_ATR(df, window=14):
     df = df.copy()
    
     df["prev_close"] = df["close"].shift(1)
    
     df["tr1"] = df["high"] - df["low"]
     df["tr2"] = (df["high"] - df["prev_close"]).abs()
     df["tr3"] = (df["low"] - df["prev_close"]).abs()
    
     df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

     df["atr"] = df["true_range"].rolling(window=window).mean()

     df.drop(columns=["prev_close", "tr1", "tr2", "tr3", "true_range"], inplace=True)
    
     return df

    @staticmethod
    def add_volume_indicators(df):
        df['on_balance_volume'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['acc_dist_index'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df


    @staticmethod
    def ta_factor_indcators(df):
        df = pd.DataFrame(df, columns=['timestamp','close', 'high', 'low', 'open', 'volume'])
        df = alpha_factor.preprocess_data(df)

        df = alpha_factor.add_momentum_indicators(df)
        df = alpha_factor.add_trend_indicators(df)
        df = alpha_factor.add_volatility_indicators(df)
        df = alpha_factor.add_volume_indicators(df)
        df = alpha_factor.add_ATR(df)

        return df

# Define helper functions for alpha formulas
def rank(series):
    return series.rank(pct=True)

def ts_rank(series, window):
    return series.rolling(int(window)).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def delta(series, period=1):
    return series.diff(period)

def delay(series, period=1):
    return series.shift(period)

def correlation(x, y, window):
    return x.rolling(int(window)).corr(y)

def covariance(x, y, window):
    return x.rolling(int(window)).cov(y)

def signed_power(series, exponent):
    return np.sign(series) * (np.abs(series) ** exponent)

def stddev(series, window):
    return pd.Series(series).rolling(int(window)).std()

def sum_(series, window):
    return series.rolling(int(window)).sum()

def ts_min(series, window):
    return pd.Series(series).rolling(int(window)).min()

def ts_max(series, window):
    return pd.Series(series).rolling(int(window)).max()

def decay_linear(series, window):
    weights = np.arange(1, int(window) + 1)
    return series.rolling(int(window)).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def scale(series):
    return series / np.sum(np.abs(series))

def product(series):
    return pd.Series(series).prod()

def sign(series):
    return np.sign(series)

def log(series):
    return np.log(series)

def sum_series(series, window):
    return series.rolling(int(window)).sum()

def Ts_Rank(series, window):
    return series.rolling(int(window)).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def IndNeutralize(series, group):
    return series.groupby(group).transform(lambda x: x - x.mean())

def min_(series, window):
    return series.rolling(int(window)).min()

def ts_argmax(series, window):
    return series.rolling(int(window)).apply(np.argmax) / window


# Example alpha formula implementation
# Alpha Factors
class Alpha_Zero:
 def __init__(self):
     super().__init__()

 @staticmethod       
 def alpha_1(df):
    condition = df['returns'] < 0
    expr = np.where(condition, stddev(df['returns'], 20), df['close'])
    ranked = rank(ts_max(signed_power(expr, 2), 5))
    return ranked - 0.5

 @staticmethod
 def alpha_2(df):
    log_volume = np.log(df['volume'].replace(0, np.nan))  # Avoid log(0)
    delta_log_vol = delta(log_volume, 2)
    ranked_delta_log_vol = rank(delta_log_vol)

    price_change = (df['close'] - df['open']) / df['open']
    ranked_price_change = rank(price_change)

    return -1 * correlation(ranked_delta_log_vol, ranked_price_change, 6)

 @staticmethod
 def alpha_3(df):
    return (-1 * correlation(rank(df['close']), rank(df['volume']), 10))

 @staticmethod
 def alpha_4(df):
    return (-1 * ts_rank(rank(df['low']), 9))

 @staticmethod
 def alpha_5(df):
    return (rank((df['open'] - df['vwap'])) * (-1 * rank(df['close'] - df['vwap']).abs()))

 @staticmethod
 def alpha_6(df):
    return (-1 * correlation(df['open'] , df['close'], 10))

 @staticmethod
 def alpha_7(df):
    adv20 = df['volume'].rolling(window=20).mean()
    delat_close_7 = delta(df['close'], 7)
    ts_r = ts_rank(abs(delat_close_7), 60)
    
    return pd.Series( np.where(
            adv20 < df['volume'],
            (-1 * ts_r * np.sign(delat_close_7)),
            -1.0
        ),
        index=df.index
    )

 @staticmethod
 def alpha_8(df):
    term = df['open'].rolling(5).sum() * df['returns'].rolling(5).sum()
    return -1 * rank(term - delay(term, 10))

 @staticmethod
 def alpha_9(df):
    delta_close = delta(df['close'], 1)
    return np.where(
        ts_min(delta_close, 5) > 0,
        delta_close,
        np.where(ts_max(delta_close, 5) < 0, delta_close, -1 * delta_close)
    )

 @staticmethod
 def alpha_10(df):
    delta_close = delta(df['close'], 1)
    cond = np.where(
        ts_min(delta_close, 4) > 0,
        delta_close,
        np.where(ts_max(delta_close, 4) < 0, delta_close, -1 * delta_close)
    )
    return rank(pd.Series(cond, index=df.index))

 @staticmethod
 def alpha_11(df):
    diff = df['vwap'] - df['close']
    return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(df['volume'], 3))

 @staticmethod
 def alpha_12(df):
    return sign(delta(df['volume'], 1)) * (-1 * delta(df['close'], 1))

 @staticmethod
 def alpha_13(df):
    return -1 * rank(covariance(rank(df['close']), rank(df['volume']), 5))

 @staticmethod
 def alpha_14(df):
    return (-1 * rank(delta(df['returns'], 3))) * correlation(df['open'], df['volume'], 10)

 @staticmethod
 def alpha_15(df):
    corrs = rank(correlation(rank(df['high']), rank(df['volume']), 3))
    return -1 * corrs.rolling(window=3).sum()

 @staticmethod
 def alpha_16(df):
    return -1 * rank(covariance(rank(df['high']), rank(df['volume']), 5))

 @staticmethod
 def alpha_17(df):
    return -1 * rank(covariance(rank(df['close']), rank(df['volume']), 5))

 @staticmethod
 def alpha_18(df):
    close_open = df['close'] - df['open']
    term = stddev(abs(close_open), 5) + close_open
    return -1 * rank(term + correlation(df['close'], df['open'], 10))

 @staticmethod
 def alpha_19(df):
    part1 = -1 * np.sign((df['close'] - df['close'].shift(7)) + df['close'].diff(7))
    rolling_sum = df['returns'].rolling(window=250).sum()  
    ranked = rolling_sum.rank(pct=True)                    
    return part1 * (1 + ranked)  
 
 @staticmethod
 def alpha_20(df):
    term = (df['close'] - delay(df['close'], 7)) + delta(df['close'], 7)
    return -1 * sign(term) * (1 + rank(1 + df['returns'].rolling(250).sum()))

 @staticmethod
 def alpha_21(df):
    avg_8 = df['close'].rolling(8).mean()
    std_8 = stddev(df['close'], 8)
    avg_2 = df['close'].rolling(2).mean()
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
    return np.where(
        (avg_8 + std_8) < avg_2, -1,
        np.where(avg_2 < (avg_8 - std_8), 1,
                 np.where((vol_ratio > 1) | (vol_ratio == 1), 1, -1))
    )

 @staticmethod
 def alpha_22(df):
    return -1 * (delta(correlation(df['high'], df['volume'], 5), 5) * rank(stddev(df['close'], 20)))

 @staticmethod
 def alpha_23(df):
    return np.where(
        (df['high'].rolling(20).mean() < df['high']),
        -1 * delta(df['high'], 2),
        0)

 @staticmethod
 def alpha_24(df):
    mean_close_100 = df['close'].rolling(100).mean()
    delta_mean = delta(mean_close_100, 100)
    delay_close = delay(df['close'], 100)
    ratio = delta_mean / delay_close
    cond = (ratio < 0.05) | (ratio == 0.05)
    return np.where(
        cond,
        -1 * (df['close'] - ts_min(df['close'], 100)),
        -1 * delta(df['close'], 3))

 @staticmethod 
 def alpha_25(df):
    adv20 = df['volume'].rolling(window=20).mean()
    return rank(((-1 * df['returns']) * adv20 * df['vwap'] * (df['high'] - df['close'])))

 @staticmethod
 def alpha_26(df):
    corr_series = correlation(rank(df['volume']), rank(df['vwap']), 6)
    sum_corr = sum_(corr_series, 2) / 2.0
    rank_sum_corr = rank(sum_corr)
    return np.where(rank_sum_corr > 0.5, -1, 1)

 @staticmethod
 def alpha_27(df):
    adv20 = df['volume'].rolling(window=20).mean()
    corr_val = correlation(adv20, df['low'], 5)
    middle = (df['high'] + df['low']) / 2
    return scale(corr_val + middle - df['close'])
 
 @staticmethod
 def alpha_28(df):
    return 
 
 @staticmethod
 def alpha_29(df):
    return -1 * ts_max(correlation(ts_rank(df['volume'], 5), ts_rank(df['high'], 5), 5), 3)

 @staticmethod
 def alpha_30(df):
    cond = (sign(df['close'] - delay(df['close'], 1)) +
            sign(delay(df['close'], 1) - delay(df['close'], 2)) +
            sign(delay(df['close'], 2) - delay(df['close'], 3)))
    return ((1.0 - rank(cond)) * sum_(df['volume'], 5)) / sum_(df['volume'], 20)

 @staticmethod
 def alpha_31(df):
    adv20 = df['close'].rolling(window=20).mean()
    part1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(df['close'], 10))), 10))))
    part2 = rank(-1 * delta(df['close'], 3))
    part3 = sign(scale(correlation(adv20, df['low'], 12)))
    return part1 + part2 + part3

 @staticmethod
 def alpha_32(df):
    part1 = scale((sum_(df['close'], 7) / 7) - df['close'])
    part2 = 20 * scale(correlation(df['vwap'], delay(df['close'], 5), 230))
    return part1 + part2
 
 @staticmethod
 def alpha_33(df):
        # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        factor = -1 * ((1 - (df['open'] / df['close'])) ** 1)
        return rank(factor)

 @staticmethod
 def alpha_34(df):
        # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        std_2 = stddev(df['returns'], 2)
        std_5 = stddev(df['returns'], 5)
        delta_1 = delta(df['close'], 1)

        rank_std_ratio = rank(std_2 / std_5)
        rank_delta_close = rank(delta_1)

        factor = (1 - rank_std_ratio) + (1 - rank_delta_close)
        return rank(factor)

 @staticmethod
 def alpha_35(df):
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        ts_rank_volume = ts_rank(df['volume'], 32)
        ts_rank_price_range = ts_rank((df['close'] + df['high'] - df['low']), 16)
        ts_rank_returns = ts_rank(df['returns'], 32)

        factor = ts_rank_volume * (1 - ts_rank_price_range) * (1 - ts_rank_returns)
        return factor

 @staticmethod
 def alpha_36(df):
    adv20 = df['volume'].rolling(window=20).mean()
    part1 = 2.21 * rank(correlation(df['close'] - df['open'], delay(df['volume'], 1), 15))
    part2 = 0.7 * rank(df['open'] - df['close'])
    part3 = 0.73 * rank(ts_rank(delay(-1 * df['returns'], 6), 5))
    part4 = rank(abs(correlation(df['vwap'], adv20, 6)))
    part5 = 0.6 * rank((sum_(df['close'], 200) / 200 - df['open']) * (df['close'] - df['open']))
    return part1 + part2 + part3 + part4 + part5

 @staticmethod
 def alpha_37(df):
        # Alpha#37: rank(correlation(delay((open - close), 1), close, 200)) + rank(open - close)
        delayed_diff = delay(df['open'] - df['close'], 1)
        corr_val = correlation(delayed_diff, df['close'], 200)
        return rank(corr_val) + rank(df['open'] - df['close'])

 @staticmethod
 def alpha_38(df):
        # Alpha#38: (-1 * rank(ts_rank(close, 10))) * rank(close / open)
        ts_rk = ts_rank(df['close'], 10)
        return (-1 * rank(ts_rk)) * rank(df['close'] / df['open'])


 @staticmethod
 def alpha_39(df):
    adv20 = df['volume'].rolling(window=20).mean()
    part1 = rank(correlation(delay(df['open'] - df['close'], 1), df['close'], 200))
    part2 = rank(df['open'] - df['close'])
    part3 = (-1 * rank(ts_rank(df['close'], 10))) * rank(df['close'] / df['open'])
    part4 = (-1 * rank(delta(df['close'], 7) * (1 - rank(decay_linear(df['volume'] / adv20, 9))))) * (1 + rank(sum_(df['returns'], 250)))
    return part1 + part2 + part3 + part4

 @staticmethod
 def alpha_40(df):
    return (-1 * rank(stddev(df['high'], 10))) * correlation(df['high'], df['volume'], 10)

 @staticmethod
 def alpha_41(df):
    return ((df['high'] * df['low']) ** 0.5) - df['vwap']

 @staticmethod
 def alpha_42(df):
    return rank(df['vwap'] - df['close']) / rank(df['vwap'] + df['close'])

 @staticmethod
 def alpha_43(df):
    adv20 = df['volume'].rolling(window=20).mean()
    
    return ts_rank(df['volume'] / adv20, 20) * ts_rank(-1 * delta(df['close'], 7), 8) 
 
 @staticmethod
 def colpute_all_alpha_factors(df_sym):
    for i in range(1, 43):
            func = getattr(Alpha_Zero, f"alpha_{i}")
            df_sym[f"alpha_factors_{i}"] = func(df_sym)
    return df_sym    
