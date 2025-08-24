import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
from ta.volume import MFIIndicator
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
from datetime import datetime
import pytz
import traceback
from scipy.stats import norm

ist = pytz.timezone("Asia/Kolkata")

def calculate_psar(high, low, close, step=0.02, max_step=0.2):
    psar = pd.Series(index=close.index)
    psar.iloc[0] = close.iloc[0]
    psarbull = pd.Series(True, index=close.index)
    psarbear = pd.Series(True, index=close.index)
    bull = True
    af = step
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]
    
    for i in range(2, len(close)):
        if bull:
            psar.iloc[i] = psar.iloc[i - 1] + af * (hp - psar.iloc[i - 1])
        else:
            psar.iloc[i] = psar.iloc[i - 1] + af * (lp - psar.iloc[i - 1])
        
        reverse = False
        
        if bull:
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                reverse = True
                psar.iloc[i] = hp
                lp = low.iloc[i]
                af = step
        else:
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                reverse = True
                psar.iloc[i] = lp
                hp = high.iloc[i]
                af = step
        
        if not reverse:
            if bull:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + step, max_step)
                if low.iloc[i - 1] < psar.iloc[i]:
                    psar.iloc[i] = low.iloc[i - 1]
                if low.iloc[i - 2] < psar.iloc[i]:
                    psar.iloc[i] = low.iloc[i - 2]
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + step, max_step)
                if high.iloc[i - 1] > psar.iloc[i]:
                    psar.iloc[i] = high.iloc[i - 1]
                if high.iloc[i - 2] > psar.iloc[i]:
                    psar.iloc[i] = high.iloc[i - 2]
        
        if bull:
            psarbull.iloc[i] = True
            psarbear.iloc[i] = False
        else:
            psarbull.iloc[i] = False
            psarbear.iloc[i] = True
    
    return psar, psarbull, psarbear

def calculate_indicators(df):
    logger.info("Calculating technical indicators...")

    try:
        # Moving Averages
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        df['RSI'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        # ATR
        df['ATR'] = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # OBV
        df['OBV'] = volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Stochastic Oscillator
        stoch = momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # VWAP
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # ROC (Rate of Change)
        df['ROC'] = momentum.ROCIndicator(df['close'], window=10).roc()
        
        # MFI (Money Flow Index)
        df['MFI'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()
        
        # Parabolic SAR
        df['SAR'], df['SAR_bullish'], df['SAR_bearish'] = calculate_psar(df['high'], df['low'], df['close'])
        
        # Aroon Indicator
        aroon = trend.AroonIndicator(df['close'], window=25)
        df['Aroon_Up'] = aroon.aroon_up()
        df['Aroon_Down'] = aroon.aroon_down()
        
        # Fibonacci Retracement Levels
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        df['Fib_23.6'] = high - 0.236 * diff
        df['Fib_38.2'] = high - 0.382 * diff
        df['Fib_50.0'] = high - 0.5 * diff
        df['Fib_61.8'] = high - 0.618 * diff
        
        # Ichimoku Cloud
        ichimoku = trend.IchimokuIndicator(df['high'], df['low'])
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()
        
        # ADX (Average Directional Index)
        adx = trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_Positive'] = adx.adx_pos()
        df['ADX_Negative'] = adx.adx_neg()
        
        # CCI (Commodity Channel Index)
        df['CCI'] = trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        
        # Williams %R
        df['Williams_R'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()

        logger.info("Technical indicators calculated successfully.")
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        logger.error(f"Full error traceback: {traceback.format_exc()}")

    # Fill NaN values with 0
    df = df.fillna(0)

    return df

def calculate_resistance_levels(df, window=20):
    logger.info("Calculating resistance levels...")
    try:
        df['resistance'] = df['high'].rolling(window=window).max()
        df['support'] = df['low'].rolling(window=window).min()
        logger.info("Resistance levels calculated successfully.")
    except Exception as e:
        logger.error(f"Error calculating resistance levels: {e}")
    return df

def calculate_option_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Implied volatility
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
    else:  # put
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, vega

def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility using the Newton-Raphson method
    """
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5  # Initial guess
    for i in range(MAX_ITERATIONS):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = calculate_option_greeks(S, K, T, r, sigma, option_type)[3]
        diff = option_price - price
        if abs(diff) < PRECISION:
            return sigma
        sigma = sigma + diff / vega
    return sigma

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price using Black-Scholes formula
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_put_call_ratio(options_chain):
    """
    Calculate Put-Call Ratio
    options_chain: DataFrame containing options data
    """
    call_volume = options_chain[options_chain['instrument_type'] == 'CE']['volume'].sum()
    put_volume = options_chain[options_chain['instrument_type'] == 'PE']['volume'].sum()
    
    if call_volume == 0:
        return float('inf')  # Avoid division by zero
    
    return put_volume / call_volume
    
def calculate_volatility(data, window=20):
    try:
        if 'last_price' in data.columns:
            price_column = 'last_price'
        elif 'underlying_price' in data.columns:
            price_column = 'underlying_price'
        elif 'close' in data.columns:
            price_column = 'close'
        else:
            logger.warning("No suitable price column found for volatility calculation")
            return 0.02  # Default volatility

        # Calculate returns
        returns = data[price_column].pct_change().dropna()

        # Calculate rolling standard deviation
        rolling_std = returns.rolling(window=window).std()

        # Annualize the volatility
        volatility = rolling_std * np.sqrt(252)  # Assuming 252 trading days in a year

        # Get the most recent volatility value
        current_volatility = volatility.iloc[-1] if not volatility.empty else 0.02

        # Log the calculated volatility
        logger.info(f"Calculated volatility: {current_volatility:.4f} using {price_column}")

        return current_volatility if not np.isnan(current_volatility) else 0.02
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        logger.exception("Full traceback:")
        return 0.02  # Default to 2% if there's an error

def adjust_parameters_based_on_volatility(volatility):
    base_buy_threshold = Config.BUY_SIGNAL_THRESHOLD
    base_sell_threshold = Config.SELL_SIGNAL_THRESHOLD

    if volatility > 0.4:  # High volatility
        return base_buy_threshold * 1.2, base_sell_threshold * 1.2
    elif volatility < 0.2:  # Low volatility
        return base_buy_threshold * 0.8, base_sell_threshold * 0.8
    else:  # Normal volatility
        return base_buy_threshold, base_sell_threshold
        
def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR) for the given data.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns
    period (int): The number of periods to use for ATR calculation (default is 14)
    
    Returns:
    pd.Series: ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate True Range
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def add_atr_to_dataframe(df, period=14):
    """
    Add ATR column to the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data
    period (int): The number of periods to use for ATR calculation (default is 14)
    
    Returns:
    pd.DataFrame: Original DataFrame with added ATR column
    """
    df['ATR'] = calculate_atr(df, period)
    return df

def is_high_volatility(data, atr_column='ATR', lookback=20, threshold=1.5):
    """
    Check if current volatility is high compared to recent average.

    Parameters:
    data (pd.DataFrame): DataFrame containing ATR data
    atr_column (str): Name of the ATR column in the DataFrame
    lookback (int): Number of periods to look back for average ATR
    threshold (float): Multiplier to determine high volatility

    Returns:
    bool: True if current volatility is high, False otherwise
    """

    if len(data) < lookback:
        logger.warning(f"Not enough data points for volatility check. Required: {lookback}, Available: {len(data)}")
        return False

    current_atr = data[atr_column].iloc[-1]
    average_atr = data[atr_column].rolling(window=lookback).mean().iloc[-1]

    is_high = current_atr > average_atr * threshold
    logger.info(f"Volatility check: Current ATR: {current_atr:.4f}, Average ATR: {average_atr:.4f}, Threshold: {threshold}, Is High: {is_high}")

    return is_high

def calculate_rsi_divergence(df, period=14):
    """
    Calculate RSI divergence.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price and RSI data
    period (int): Number of periods for divergence calculation
    
    Returns:
    tuple: (bullish_divergence, bearish_divergence)
    """
    price_low = df['low'].rolling(window=period).min()
    price_high = df['high'].rolling(window=period).max()
    rsi_low = df['RSI'].rolling(window=period).min()
    rsi_high = df['RSI'].rolling(window=period).max()

    bullish_divergence = (price_low.iloc[-1] < price_low.iloc[-period]) and (rsi_low.iloc[-1] > rsi_low.iloc[-period])
    bearish_divergence = (price_high.iloc[-1] > price_high.iloc[-period]) and (rsi_high.iloc[-1] < rsi_high.iloc[-period])

    return bullish_divergence, bearish_divergence

def calculate_volume_profile(df, price_levels=10):
    """
    Calculate Volume Profile.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price and volume data
    price_levels (int): Number of price levels to divide the range into
    
    Returns:
    pd.DataFrame: Volume profile data
    """
    price_range = df['high'].max() - df['low'].min()
    level_height = price_range / price_levels

    volume_profile = []
    for i in range(price_levels):
        level_low = df['low'].min() + i * level_height
        level_high = level_low + level_height
        level_volume = df[(df['close'] >= level_low) & (df['close'] < level_high)]['volume'].sum()
        volume_profile.append({'price_level': (level_low + level_high) / 2, 'volume': level_volume})

    return pd.DataFrame(volume_profile)

def calculate_pivot_points(df):
    """
    Calculate Pivot Points.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing OHLC data
    
    Returns:
    dict: Pivot points
    """
    pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
    r1 = 2 * pivot - df['low'].iloc[-1]
    s1 = 2 * pivot - df['high'].iloc[-1]
    r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
    s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])

    return {
        'pivot': pivot,
        'r1': r1,
        's1': s1,
        'r2': r2,
        's2': s2
    }

def calculate_support_resistance(df, window=20):
    """
    Calculate dynamic support and resistance levels.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing OHLC data
    window (int): Look-back period for calculating levels
    
    Returns:
    tuple: (support, resistance)
    """
    support = df['low'].rolling(window=window).min()
    resistance = df['high'].rolling(window=window).max()
    return support.iloc[-1], resistance.iloc[-1]

def identify_chart_patterns(df, window=20):
    """
    Identify common chart patterns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing OHLC data
    window (int): Look-back period for pattern identification
    
    Returns:
    dict: Identified patterns
    """
    patterns = {}

    # Head and Shoulders
    left_shoulder = df['high'].iloc[-window:-window//2].max()
    head = df['high'].iloc[-window:].max()
    right_shoulder = df['high'].iloc[-window//2:].max()
    neckline = min(df['low'].iloc[-window:-window//2].min(), df['low'].iloc[-window//2:].min())

    if left_shoulder < head and right_shoulder < head and abs(left_shoulder - right_shoulder) / head < 0.1:
        patterns['head_and_shoulders'] = True

    # Double Top
    tops = df['high'].iloc[-window:].nlargest(2)
    if abs(tops.iloc[0] - tops.iloc[1]) / tops.iloc[0] < 0.02:
        patterns['double_top'] = True

    # Double Bottom
    bottoms = df['low'].iloc[-window:].nsmallest(2)
    if abs(bottoms.iloc[0] - bottoms.iloc[1]) / bottoms.iloc[0] < 0.02:
        patterns['double_bottom'] = True

    return patterns

def calculate_momentum_indicators(df):
    """
    Calculate additional momentum indicators.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data
    
    Returns:
    pd.DataFrame: DataFrame with added momentum indicators
    """
    # Rate of Change (ROC)
    df['ROC'] = df['close'].pct_change(periods=10) * 100

    # Momentum
    df['Momentum'] = df['close'] - df['close'].shift(10)

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    return df

def identify_candlestick_patterns(df):
    """
    Identify common candlestick patterns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing OHLC data
    
    Returns:
    dict: Identified candlestick patterns
    """
    patterns = {}

    # Doji
    doji_threshold = 0.1
    df['body'] = abs(df['close'] - df['open'])
    df['wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['tail'] = df[['open', 'close']].min(axis=1) - df['low']

    patterns['doji'] = (df['body'] <= doji_threshold * (df['high'] - df['low'])).iloc[-1]

    # Hammer
    patterns['hammer'] = (df['tail'] > 2 * df['body']) & (df['wick'] < df['body']).iloc[-1]

    # Shooting Star
    patterns['shooting_star'] = (df['wick'] > 2 * df['body']) & (df['tail'] < df['body']).iloc[-1]

    # Engulfing
    patterns['bullish_engulfing'] = (df['open'].iloc[-2] > df['close'].iloc[-2]) & \
                                    (df['close'].iloc[-1] > df['open'].iloc[-2]) & \
                                    (df['open'].iloc[-1] < df['close'].iloc[-2])

    patterns['bearish_engulfing'] = (df['close'].iloc[-2] > df['open'].iloc[-2]) & \
                                    (df['open'].iloc[-1] > df['close'].iloc[-2]) & \
                                    (df['close'].iloc[-1] < df['open'].iloc[-2])

    return patterns

def calculate_market_breadth(index_data, stock_data):
    """
    Calculate market breadth indicators.
    
    Parameters:
    index_data (pd.DataFrame): DataFrame containing index data
    stock_data (dict): Dictionary of DataFrames containing individual stock data
    
    Returns:
    dict: Market breadth indicators
    """
    advancing = sum(1 for stock in stock_data.values() if stock['close'].iloc[-1] > stock['close'].iloc[-2])
    declining = sum(1 for stock in stock_data.values() if stock['close'].iloc[-1] < stock['close'].iloc[-2])

    breadth = {
        'advance_decline_ratio': advancing / declining if declining != 0 else float('inf'),
        'percent_above_sma50': sum(1 for stock in stock_data.values() if stock['close'].iloc[-1] > stock['SMA50'].iloc[-1]) / len(stock_data) * 100,
        'percent_above_sma200': sum(1 for stock in stock_data.values() if stock['close'].iloc[-1] > stock['SMA200'].iloc[-1]) / len(stock_data) * 100,
    }

    return breadth

def calculate_option_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate
    sigma (float): Implied volatility
    option_type (str): 'call' or 'put'
    
    Returns:
    tuple: (delta, gamma, theta, vega)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        delta = -norm.cdf(-d1)
        theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, vega

# Add any additional technical indicator functions as needed

if __name__ == "__main__":
    # You can add some test code here to run the indicator calculations independently
    pass