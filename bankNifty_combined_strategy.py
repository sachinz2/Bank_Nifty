from logging_utils import banknifty_logger as logger
from bankNifty_ML_model import predict_ml_model
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from bankNifty_config import Config
from ta import trend as ta_trend
from bankNifty_technical_indicators import calculate_indicators, calculate_resistance_levels, add_atr_to_dataframe, is_high_volatility, calculate_atr, calculate_volatility
from bankNifty_feature_engineering import engineer_features, preprocess_data
import json

ist = pytz.timezone("Asia/Kolkata")

def combined_strategy(option_data, historical_data, models, features, options_analysis, market_trend, data_fetcher):
    logger.info("Entering combined_strategy function")
    
    try:
        # Process option_data
        if isinstance(option_data, pd.DataFrame):
            logger.info(f"Received option_data as DataFrame with shape: {option_data.shape}")
            option_df = option_data
        elif isinstance(option_data, dict):
            logger.info("Received option_data as dictionary")
            option_df = pd.DataFrame([option_data])
        else:
            logger.error(f"Unexpected option_data type: {type(option_data)}")
            return False, False, 'neutral'

        # Ensure historical_data is a DataFrame
        if not isinstance(historical_data, pd.DataFrame):
            logger.error(f"Unexpected historical_data type: {type(historical_data)}")
            return False, False, 'neutral'

        logger.info(f"Received market trend: {market_trend}")
        
        # Log original columns for debugging
        logger.info(f"Original option data columns: {option_df.columns.tolist()}")
        logger.info(f"Original historical data columns: {historical_data.columns.tolist()}")

        # Combine option data with latest historical indicators
        latest_historical = historical_data.iloc[-1:].reset_index(drop=True)
        df = pd.concat([option_df, latest_historical], axis=1)

        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        logger.info(f"Combined DataFrame shape: {df.shape}")
        logger.info(f"Combined DataFrame columns: {df.columns.tolist()}")

        # Check if required columns are present in the combined DataFrame
        required_columns = ['RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'VWAP', 'ATR', 
                            'last_price', 'strike', 'underlying_price', 'implied_volatility', 'delta', 'theta']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in combined data: {missing_columns}")
            return False, False, 'neutral'
        
        # Use 'last_price' instead of 'close' for recent price change calculation
        try:
            recent_price_change = (df['last_price'].iloc[-1] - df['underlying_price'].iloc[-1]) / df['underlying_price'].iloc[-1]
            logger.info(f"Recent price change: {recent_price_change:.2%}")
        except IndexError:
            logger.error("DataFrame is empty or doesn't have enough rows")
            return False, False, 'neutral'

        volatility = calculate_volatility(df)
        logger.info(f"Current volatility: {volatility:.4f}")
        buy_threshold, sell_threshold = adjust_parameters_based_on_volatility(volatility)

        # Adjust thresholds based on market trend
        if 'downtrend' in market_trend.lower():
            buy_threshold *= 1.2  # Increase buy threshold in downtrend
            sell_threshold *= 0.8  # Decrease sell threshold in downtrend
            logger.info(f"Adjusted thresholds for downtrend trend or significant price change. New Buy Threshold: {buy_threshold:.2f}, New Sell Threshold: {sell_threshold:.2f}")
            
        # Adjust thresholds based on market trend and recent price change
        if 'strong' in market_trend or abs(recent_price_change) > 0.01:
            buy_threshold *= 0.8
            sell_threshold *= 0.8
            logger.info(f"Adjusted thresholds for strong trend or significant price change. New Buy Threshold: {buy_threshold:.2f}, New Sell Threshold: {sell_threshold:.2f}")
        
        if is_high_volatility(df):
            buy_threshold *= 1.1
            sell_threshold *= 1.1
            logger.info(f"Adjusted thresholds for high volatility. New Buy Threshold: {buy_threshold:.2f}, New Sell Threshold: {sell_threshold:.2f}")
    
        # Fetch additional data for feature engineering
        options_chain_data = data_fetcher.fetch_options_chain_data()
        sentiment_data = data_fetcher.get_market_sentiment_indicators()
        market_breadth_data = data_fetcher.fetch_market_breadth_data()
        volatility_data = data_fetcher.fetch_volatility_data()

        # Engineer features
        feature_data = engineer_features(df, options_chain_data, sentiment_data, market_breadth_data, volatility_data)
        
        # Preprocess data
        preprocessed_data = preprocess_data(feature_data)

        # Use XGBoost model for predictions
        ml_buy, ml_sell, xgb_prediction = predict_ml_model(models, features, preprocessed_data)
        
        # Technical signals
        technical_signals = [
            (trend_following_strategy(df), 1.2),
            (rsi_strategy(df), 0.8),
            (bollinger_bands_strategy(df), 0.7),
            (breakout_strategy(df), 1.0),
            (fibonacci_strategy(df), 0.5),
            (momentum_strategy(df), 1.0),
        ]
        
        options_buy, options_sell = options_analysis_strategy(options_analysis)
        
        weights = adjust_weights(market_trend)
        options_weight = 2.0 if 'uptrend' in market_trend else 1.5
        trend_weight = 2.0 if 'strong' in market_trend else 1.5
        
        # Calculate technical signals with weights
        tech_buy_signals = sum(signal[0] * weight for signal, weight in technical_signals if signal[0])
        tech_sell_signals = sum(signal[1] * weight for signal, weight in technical_signals if signal[1])

        # Include ML model predictions, options analysis, and market trend
        buy_signals = (tech_buy_signals + 
                       ml_buy * Config.ML_WEIGHT * 1.5 + 
                       options_buy * options_weight +
                       (market_trend in ['uptrend', 'strong_uptrend']) * trend_weight)
        
        sell_signals = (tech_sell_signals + 
                        ml_sell * Config.ML_WEIGHT * 1.5 + 
                        options_sell * options_weight +
                        (market_trend in ['downtrend', 'strong_downtrend']) * trend_weight)

        total_signals = sum(weight for _, weight in technical_signals) + Config.ML_WEIGHT * 1.5 + options_weight + trend_weight
        
        # Adjust weights based on recent price change
        if recent_price_change > 0.005:
            weights['bullish'] *= 1.2
            weights['bearish'] *= 0.8
        elif recent_price_change < -0.005:
            weights['bullish'] *= 0.8
            weights['bearish'] *= 1.2

        buy_ratio = (buy_signals / total_signals) * weights['bullish']
        sell_ratio = (sell_signals / total_signals) * weights['bearish']
        
        logger.info(f"ML signals: Buy={ml_buy}, Sell={ml_sell}")
        logger.info(f"Options signals: Buy={options_buy}, Sell={options_sell}")
        logger.info(f"Technical signals: {[signal for signal in technical_signals]}")
        logger.info(f"Market trend: {market_trend}")
        logger.info(f"Buy ratio: {buy_ratio:.2f}, Sell ratio: {sell_ratio:.2f}")

        # Generate signals
        buy_signal = buy_ratio >= buy_threshold
        sell_signal = sell_ratio >= sell_threshold

        # Determine overall market direction
        options_sentiment = options_analysis.get('chain_analysis', {}).get('overall_sentiment', 'neutral')
        
        if recent_price_change > 0.005 or market_trend in ['uptrend', 'strong_uptrend']:
            market_direction = 'bullish'
        elif recent_price_change < -0.005 or market_trend in ['downtrend', 'strong_downtrend']:
            market_direction = 'bearish'
        else:
            market_direction = 'neutral'

        # Final decision logic
        if buy_signal and not sell_signal and market_direction != 'bearish':
            final_decision = 'buy'
        elif sell_signal and not buy_signal and market_direction != 'bullish':
            final_decision = 'sell'
        else:
            final_decision = 'hold'

        logger.info(f"Final decision: {final_decision}")
        logger.info(f"Buy ratio: {buy_ratio:.2f}, Sell ratio: {sell_ratio:.2f}")
        logger.info(f"Adjusted thresholds: Buy={buy_threshold:.2f}, Sell={sell_threshold:.2f}")
        logger.info(f"Final Buy signal: {buy_signal}, Sell signal: {sell_signal}")     
        logger.info(f"Options chain sentiment: {options_sentiment}")
        logger.info(f"Overall market direction: {market_direction}")
        
        return buy_signal, sell_signal, market_direction
    except Exception as e:
        logger.error(f"Error generating combined strategy signals: {e}")
        logger.exception("Full traceback:")
        return False, False, 'neutral'

def adjust_weights(market_trend):
    if market_trend == 'strong_uptrend':
        return {'bullish': 1.4, 'bearish': 0.6}
    elif market_trend == 'uptrend':
        return {'bullish': 1.3, 'bearish': 0.7}
    elif market_trend == 'strong_downtrend':
        return {'bullish': 0.6, 'bearish': 1.4}
    elif market_trend == 'downtrend':
        return {'bullish': 0.7, 'bearish': 1.3}
    else:  # consolidation or unknown
        return {'bullish': 1.0, 'bearish': 1.0}

def trend_following_strategy(df):
    if len(df) < 2:
        return False, False
    current_price = df['last_price'].iloc[-1]
    prev_price = df['last_price'].iloc[-2]
    sma20 = df['SMA20'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    sma200 = df['SMA200'].iloc[-1]
    
    buy_signal = (current_price > sma20 > sma50 > sma200) and (current_price > prev_price)
    sell_signal = (current_price < sma20 < sma50 < sma200) and (current_price < prev_price)
    
    return buy_signal, sell_signal
    
def rsi_strategy(df):
    rsi = df['RSI'].iloc[-1]
    buy_signal = (rsi < 35)
    sell_signal = (rsi > 65)
    return buy_signal, sell_signal

def macd_strategy(df):
    macd = df['MACD'].iloc[-1]
    signal = df['MACD_Signal'].iloc[-1]
    buy_signal = (macd > signal) and (macd < 0)
    sell_signal = (macd < signal) and (macd > 0)
    return buy_signal, sell_signal
    
def bollinger_bands_strategy(df):
    current_price = df['last_price'].iloc[-1]
    bb_high = df['BB_High'].iloc[-1]
    bb_low = df['BB_Low'].iloc[-1]
    buy_signal = current_price < bb_low
    sell_signal = current_price > bb_high
    return buy_signal, sell_signal

def vwap_strategy(df):
    current_price = df['last_price'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    buy_signal = current_price > vwap
    sell_signal = current_price < vwap
    return buy_signal, sell_signal

def breakout_strategy(df):
    current_price = df['last_price'].iloc[-1]
    resistance = df['resistance'].iloc[-1]
    support = df['support'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    buy_signal = (current_price > resistance + 0.5 * atr)
    sell_signal = (current_price < support - 0.5 * atr)
    return buy_signal, sell_signal
    
def fibonacci_strategy(df):
    current_price = df['last_price'].iloc[-1]
    fib_38_2 = df['Fib_38.2'].iloc[-1]
    fib_61_8 = df['Fib_61.8'].iloc[-1]
    buy_signal = current_price > fib_61_8
    sell_signal = current_price < fib_38_2
    return buy_signal, sell_signal

def momentum_strategy(df):
    roc = df['ROC'].iloc[-1]
    mfi = df['MFI'].iloc[-1]
    buy_signal = roc > 0 and mfi < 30
    sell_signal = roc < 0 and mfi > 70
    return buy_signal, sell_signal

def options_analysis_strategy(options_analysis):
    logger.info("Entering options_analysis_strategy function")
    try:
        if options_analysis is None or not isinstance(options_analysis, dict):
            logger.warning("Options analysis data is not available or invalid.")
            return False, False

        put_call_ratio = options_analysis.get('put_call_ratio', 1)
        avg_call_iv = options_analysis.get('avg_call_iv', 0)
        avg_put_iv = options_analysis.get('avg_put_iv', 0)
        max_pain = options_analysis.get('max_pain', 0)
        chain_analysis = options_analysis.get('chain_analysis', {})
        iv_skew = options_analysis.get('iv_skew', 0)
        atm_pcr = options_analysis.get('atm_pcr', {})

        # Implement your options analysis logic here
        buy_signal = (put_call_ratio < 0.8 and avg_call_iv < avg_put_iv and iv_skew < -0.1)
        sell_signal = (put_call_ratio > 1.2 and avg_call_iv > avg_put_iv and iv_skew > 0.1)

        logger.info(f"Options analysis signals - Buy: {buy_signal}, Sell: {sell_signal}")
        return buy_signal, sell_signal

    except Exception as e:
        logger.error(f"Error in options analysis strategy: {e}")
        logger.exception("Full traceback:")
        return False, False
        
def adjust_parameters_based_on_volatility(volatility):
    base_buy_threshold = Config.BUY_SIGNAL_THRESHOLD
    base_sell_threshold = Config.SELL_SIGNAL_THRESHOLD

    if volatility > 0.3:  # High volatility
        return base_buy_threshold * 1.2, base_sell_threshold * 1.2
    elif volatility < 0.1:  # Low volatility
        return base_buy_threshold * 0.8, base_sell_threshold * 0.8
    else:  # Normal volatility
        return base_buy_threshold, base_sell_threshold

def analyze_market_trend(df):
    try:
        recent_df = df.iloc[-48:]  # Last 4 hours (assuming 5-minute intervals)
        very_recent_df = df.iloc[-12:]  # Last 1 hour
        
        close = recent_df['close']
        open_price = recent_df['open'].iloc[0]
        current_price = close.iloc[-1]
        
        total_change = (current_price - open_price) / open_price
        short_term_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
        very_short_term_change = (very_recent_df['close'].iloc[-1] - very_recent_df['close'].iloc[0]) / very_recent_df['close'].iloc[0]
        
        # Determine trend strength based on both 4-hour and 1-hour changes
        if total_change > 0.02 or very_short_term_change > 0.01:
            trend_strength = 'strong_uptrend'
        elif total_change > 0.01 or very_short_term_change > 0.005:
            trend_strength = 'uptrend'
        elif total_change < -0.02 or very_short_term_change < -0.01:
            trend_strength = 'strong_downtrend'
        elif total_change < -0.01 or very_short_term_change < -0.005:
            trend_strength = 'downtrend'
        elif abs(short_term_change) < 0.005 and abs(very_short_term_change) < 0.0025:
            trend_strength = 'consolidation'
        elif short_term_change > 0:
            trend_strength = 'weak_uptrend'
        elif short_term_change < 0:
            trend_strength = 'weak_downtrend'
        else:
            trend_strength = 'neutral'

        logger.info(f"Analyzed market trend: {trend_strength} (Total change: {total_change:.2%}, 4-hour change: {short_term_change:.2%}, 1-hour change: {very_short_term_change:.2%})")
        return trend_strength
    except Exception as e:
        logger.error(f"Error analyzing market trend: {e}")
        logger.exception("Full traceback:")
        return 'unknown'

def analyze_market_sentiment(data_fetcher):
    try:
        # Fetch data for the last trading day
        banknifty_data = data_fetcher.fetch_banknifty_index_data(interval='5minute', days=1)
        
        if banknifty_data.empty:
            logger.warning("No data available for market sentiment analysis.")
            return 'neutral'

        # Calculate indicators
        banknifty_data = calculate_indicators(banknifty_data)

        # Focus on more recent data (last 1 hour)
        recent_data = banknifty_data.iloc[-12:]  # Assuming 5-minute intervals, 12 * 5 = 60 minutes

        open_price = recent_data['open'].iloc[0]
        close_price = recent_data['close'].iloc[-1]
        
        # Calculate intraday change
        intraday_change = (close_price - banknifty_data['open'].iloc[0]) / banknifty_data['open'].iloc[0]
        recent_change = (close_price - open_price) / open_price
        
        # Determine sentiment based primarily on price change
        if intraday_change > 0.01 or recent_change > 0.005:  # 1% intraday or 0.5% in last hour
            sentiment = 'bullish'
        elif intraday_change < -0.01 or recent_change < -0.005:  # -1% intraday or -0.5% in last hour
            sentiment = 'bearish'
        else:
            # If the price change is not significant, consider other factors
            rsi = recent_data['RSI'].iloc[-1]
            macd = recent_data['MACD'].iloc[-1]
            macd_signal = recent_data['MACD_Signal'].iloc[-1]
            
            sentiment_score = 0
            
            # RSI
            if rsi > 60:
                sentiment_score += 1
            elif rsi < 40:
                sentiment_score -= 1
            
            # MACD
            if macd > macd_signal:
                sentiment_score += 1
            else:
                sentiment_score -= 1
            
            # Volume
            avg_volume = recent_data['volume'].mean()
            if recent_data['volume'].iloc[-1] > avg_volume * 1.2:  # 20% above average
                if recent_change > 0:
                    sentiment_score += 1
                else:
                    sentiment_score -= 1
            
            # Determine final sentiment
            if sentiment_score > 0:
                sentiment = 'bullish'
            elif sentiment_score < 0:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'

        logger.info(f"Market sentiment analysis: Sentiment={sentiment}, Intraday Change={intraday_change:.2%}, Recent Change={recent_change:.2%}")
        return sentiment

    except Exception as e:
        logger.error(f"Error analyzing market sentiment: {e}")
        logger.exception("Full traceback:")
        return 'neutral'  # Default to neutral in case of error

def get_market_trend(historical_data):
    try:
        last_price = historical_data['close'].iloc[-1]
        sma_20 = historical_data['SMA20'].iloc[-1]
        sma_50 = historical_data['SMA50'].iloc[-1]
        rsi = historical_data['RSI'].iloc[-1]

        if last_price > sma_20 > sma_50 and rsi > 60:
            return "strong_uptrend"
        elif last_price > sma_20 and last_price > sma_50 and rsi > 50:
            return "uptrend"
        elif last_price < sma_20 < sma_50 and rsi < 40:
            return "strong_downtrend"
        elif last_price < sma_20 and last_price < sma_50 and rsi < 50:
            return "downtrend"
        else:
            return "sideways"
    except Exception as e:
        logger.error(f"Error determining market trend: {e}")
        return "unknown"

# You can add more helper functions here if needed

if __name__ == "__main__":
    # You can add some test code here to run the strategies independently
    pass