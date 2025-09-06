import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Calculates all the features required for the trading model.
    """

    def __init__(self, lookback=1000):
        self.lookback = lookback

    def _z_score_normalization(self, series):
        return (series - series.rolling(window=self.lookback).mean()) / series.rolling(window=self.lookback).std()

    def add_features(self, df):
        """
        Adds all features to the input DataFrame.

        Args:
            df: A pandas DataFrame with OHLCV data.

        Returns:
            A pandas DataFrame with the added features.
        """
        # Resample and forward-fill
        df = df.resample('1min').ffill()

        # OHLCV raw (last 60 bars) - this will be handled by the data loader

        # EMAs
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.max([high_low, high_close, low_close], axis=0)
        df['atr'] = pd.Series(tr).rolling(window=14).mean()

        # Bollinger Band Width
        ma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_width'] = (ma_20 + 2 * std_20) - (ma_20 - 2 * std_20)

        # Volume delta ratio
        df['volume_delta_ratio'] = df['volume'] / df['volume'].rolling(window=60).mean()

        # Mid-price return features
        mid_price = (df['high'] + df['low']) / 2
        df['mid_price_return_1min'] = mid_price.pct_change()
        df['mid_price_return_3min_avg'] = df['mid_price_return_1min'].rolling(window=3).mean()

        # Normalization
        feature_cols = ['ema_5', 'ema_15', 'ema_60', 'rsi', 'atr', 'bb_width', 'volume_delta_ratio', 'mid_price_return_1min', 'mid_price_return_3min_avg']
        for col in feature_cols:
            df[col + '_norm'] = self._z_score_normalization(df[col])

        # Drop rows with NaN values created by rolling windows
        df.dropna(inplace=True)

        return df
