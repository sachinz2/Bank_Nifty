import pandas as pd
import numpy as np
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
from ta import trend, momentum, volatility
import pytz

ist = pytz.timezone('Asia/Kolkata')

class MarketAnalyzer:
    def __init__(self):
        self.prev_analysis = None
        self.trend_change_count = 0
        self.last_trend = None

    def analyze_market_conditions(self, df, options_data, sentiment_data):
        """
        Comprehensive market analysis combining various indicators
        """
        try:
            # Technical Analysis
            trend_analysis = self.analyze_trend(df)
            momentum_analysis = self.analyze_momentum(df)
            volatility_analysis = self.analyze_volatility(df)
            
            # Options Analysis
            options_analysis = self.analyze_options_data(options_data)
            
            # Volume Analysis
            volume_analysis = self.analyze_volume(df)
            
            # Sentiment Analysis
            sentiment_analysis = self.analyze_sentiment(sentiment_data)

            # Combine all analyses
            market_condition = self.combine_analyses(
                trend_analysis,
                momentum_analysis,
                volatility_analysis,
                options_analysis,
                volume_analysis,
                sentiment_analysis
            )

            self.update_trend_tracking(market_condition['trend'])

            return market_condition

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            logger.exception("Full traceback:")
            return {'trend': 'unknown', 'strength': 0, 'volatility': 'normal'}

    def analyze_trend(self, df):
        """
        Analyze market trend using multiple indicators
        """
        try:
            # Calculate various trend indicators
            sma_20 = df['SMA20'].iloc[-1]
            sma_50 = df['SMA50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            adx = df['ADX'].iloc[-1]
            
            # Determine trend direction
            trend_direction = 'uptrend' if current_price > sma_20 > sma_50 else 'downtrend'
            
            # Determine trend strength
            trend_strength = 'strong' if adx > Config.ADX_THRESHOLD else 'weak'
            
            # Calculate trend momentum
            momentum = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'momentum': momentum,
                'adx': adx
            }

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'direction': 'unknown', 'strength': 'unknown', 'momentum': 0, 'adx': 0}

    def analyze_momentum(self, df):
        """
        Analyze market momentum
        """
        try:
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            
            momentum_signals = {
                'rsi_signal': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish',
                'rsi': rsi,
                'macd': macd
            }
            
            return momentum_signals

        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {'rsi_signal': 'neutral', 'macd_signal': 'neutral', 'rsi': 50, 'macd': 0}

    def analyze_volatility(self, df):
        """
        Analyze market volatility
        """
        try:
            atr = df['ATR'].iloc[-1]
            bb_width = (df['BB_High'].iloc[-1] - df['BB_Low'].iloc[-1]) / df['BB_Mid'].iloc[-1]
            
            # Calculate historical volatility
            returns = df['close'].pct_change()
            hist_vol = returns.std() * np.sqrt(252)
            
            volatility_state = 'high' if hist_vol > Config.HIGH_VOLATILITY_THRESHOLD else 'low' if hist_vol < Config.LOW_VOLATILITY_THRESHOLD else 'normal'
            
            return {
                'state': volatility_state,
                'atr': atr,
                'bb_width': bb_width,
                'historical_volatility': hist_vol
            }

        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {'state': 'normal', 'atr': 0, 'bb_width': 0, 'historical_volatility': 0}

    def analyze_options_data(self, options_data):
        """
        Analyze options market data
        """
        try:
            if not options_data:
                return {'sentiment': 'neutral', 'iv_skew': 0}

            pcr = options_data.get('put_call_ratio', 1)
            iv_skew = options_data.get('iv_skew', 0)
            
            sentiment = 'bullish' if pcr < 0.8 else 'bearish' if pcr > 1.2 else 'neutral'
            
            return {
                'sentiment': sentiment,
                'pcr': pcr,
                'iv_skew': iv_skew
            }

        except Exception as e:
            logger.error(f"Error in options analysis: {e}")
            return {'sentiment': 'neutral', 'pcr': 1, 'iv_skew': 0}

    def analyze_volume(self, df):
        """
        Analyze volume patterns
        """
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            volume_trend = 'high' if current_volume > avg_volume * 1.5 else 'low' if current_volume < avg_volume * 0.5 else 'normal'
            
            return {
                'trend': volume_trend,
                'current_volume': current_volume,
                'average_volume': avg_volume
            }

        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {'trend': 'normal', 'current_volume': 0, 'average_volume': 0}

    def analyze_sentiment(self, sentiment_data):
        """
        Analyze market sentiment
        """
        try:
            if not sentiment_data:
                return {'sentiment': 'neutral', 'strength': 0}

            sentiment = sentiment_data.get('market_sentiment', 'neutral')
            vix = sentiment_data.get('vix', 15)
            
            sentiment_strength = 'strong' if vix > 20 else 'weak'
            
            return {
                'sentiment': sentiment,
                'strength': sentiment_strength,
                'vix': vix
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'strength': 'weak', 'vix': 15}

    def combine_analyses(self, trend, momentum, volatility, options, volume, sentiment):
        """
        Combine all analyses into a single market condition assessment
        """
        try:
            # Determine overall trend
            if trend['direction'] == 'uptrend' and trend['strength'] == 'strong':
                overall_trend = 'strong_uptrend'
            elif trend['direction'] == 'uptrend' and trend['strength'] == 'weak':
                overall_trend = 'weak_uptrend'
            elif trend['direction'] == 'downtrend' and trend['strength'] == 'strong':
                overall_trend = 'strong_downtrend'
            elif trend['direction'] == 'downtrend' and trend['strength'] == 'weak':
                overall_trend = 'weak_downtrend'
            else:
                overall_trend = 'consolidation'

            # Calculate overall strength score (0-100)
            strength_score = self.calculate_strength_score(
                trend, momentum, volatility, options, volume, sentiment
            )

            # Determine trading conditions
            trading_conditions = self.assess_trading_conditions(
                trend, momentum, volatility, options, volume, sentiment
            )

            market_condition = {
                'trend': overall_trend,
                'strength': strength_score,
                'volatility': volatility['state'],
                'momentum': momentum['rsi_signal'],
                'volume_trend': volume['trend'],
                'sentiment': sentiment['sentiment'],
                'trading_conditions': trading_conditions,
                'options_sentiment': options['sentiment'],
                'timestamp': pd.Timestamp.now(tz=ist)
            }

            self.prev_analysis = market_condition
            return market_condition

        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            logger.exception("Full traceback:")
            return {
                'trend': 'unknown',
                'strength': 0,
                'volatility': 'normal',
                'momentum': 'neutral',
                'volume_trend': 'normal',
                'sentiment': 'neutral',
                'trading_conditions': 'neutral',
                'options_sentiment': 'neutral',
                'timestamp': pd.Timestamp.now(tz=ist)
            }

    def calculate_strength_score(self, trend, momentum, volatility, options, volume, sentiment):
        """
        Calculate overall market strength score (0-100)
        """
        try:
            score = 50  # Start at neutral

            # Trend component (±20)
            if trend['direction'] == 'uptrend':
                score += 10 if trend['strength'] == 'strong' else 5
            elif trend['direction'] == 'downtrend':
                score -= 10 if trend['strength'] == 'strong' else 5

            # Momentum component (±15)
            if momentum['rsi_signal'] == 'overbought':
                score += 15
            elif momentum['rsi_signal'] == 'oversold':
                score -= 15

            # Options component (±10)
            if options['sentiment'] == 'bullish':
                score += 10
            elif options['sentiment'] == 'bearish':
                score -= 10

            # Volume component (±5)
            if volume['trend'] == 'high':
                score += 5 if trend['direction'] == 'uptrend' else -5

            # Sentiment component (±10)
            if sentiment['sentiment'] == 'bullish':
                score += 10 if sentiment['strength'] == 'strong' else 5
            elif sentiment['sentiment'] == 'bearish':
                score -= 10 if sentiment['strength'] == 'strong' else 5

            # Ensure score stays within 0-100
            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error calculating strength score: {e}")
            return 50

    def assess_trading_conditions(self, trend, momentum, volatility, options, volume, sentiment):
        """
        Assess overall trading conditions
        """
        try:
            conditions = []

            # Check for strong trend
            if trend['strength'] == 'strong' and trend['adx'] > Config.ADX_THRESHOLD:
                conditions.append('trending')

            # Check for high volatility
            if volatility['state'] == 'high':
                conditions.append('volatile')

            # Check for momentum
            if momentum['rsi_signal'] in ['overbought', 'oversold']:
                conditions.append('momentum')

            # Check for volume confirmation
            if volume['trend'] == 'high':
                conditions.append('volume_confirmed')

            # Check for option signals
            if options['sentiment'] != 'neutral':
                conditions.append('options_active')

            # Determine overall conditions
            if not conditions:
                return 'neutral'
            elif 'volatile' in conditions and len(conditions) > 2:
                return 'high_risk'
            elif len(conditions) >= 3:
                return 'favorable'
            else:
                return 'moderate'

        except Exception as e:
            logger.error(f"Error assessing trading conditions: {e}")
            return 'neutral'

    def update_trend_tracking(self, current_trend):
        """
        Track trend changes and their frequency
        """
        try:
            if self.last_trend and current_trend != self.last_trend:
                self.trend_change_count += 1
                logger.info(f"Trend changed from {self.last_trend} to {current_trend}")

            self.last_trend = current_trend

        except Exception as e:
            logger.error(f"Error updating trend tracking: {e}")

    def get_trend_stability(self):
        """
        Calculate trend stability based on number of changes
        """
        try:
            if self.trend_change_count <= 2:
                return 'stable'
            elif self.trend_change_count <= 5:
                return 'moderate'
            else:
                return 'unstable'
        except Exception as e:
            logger.error(f"Error getting trend stability: {e}")
            return 'unknown'

    def should_trade(self, market_condition):
        """
        Determine if current market conditions are suitable for trading
        """
        try:
            # Check for extreme conditions
            if market_condition['volatility'] == 'high' and market_condition['strength'] < 40:
                return False

            # Check for favorable conditions
            if market_condition['trading_conditions'] == 'favorable' and \
               market_condition['volume_trend'] == 'high' and \
               market_condition['trend'] in ['strong_uptrend', 'strong_downtrend']:
                return True

            # Check for unfavorable conditions
            if market_condition['trading_conditions'] == 'high_risk' or \
               self.get_trend_stability() == 'unstable':
                return False

            # Default to moderate conditions
            return market_condition['strength'] > 60

        except Exception as e:
            logger.error(f"Error in should_trade assessment: {e}")
            return False

    def get_recommended_position_size(self, market_condition, base_size):
        """
        Adjust position size based on market conditions
        """
        try:
            multiplier = 1.0

            # Adjust based on trend strength
            if market_condition['strength'] > 80:
                multiplier *= 1.2
            elif market_condition['strength'] < 40:
                multiplier *= 0.8

            # Adjust based on volatility
            if market_condition['volatility'] == 'high':
                multiplier *= 0.7
            elif market_condition['volatility'] == 'low':
                multiplier *= 1.1

            # Adjust based on trading conditions
            if market_condition['trading_conditions'] == 'favorable':
                multiplier *= 1.2
            elif market_condition['trading_conditions'] == 'high_risk':
                multiplier *= 0.6

            recommended_size = int(base_size * multiplier)
            
            # Ensure within limits
            recommended_size = max(min(recommended_size, Config.MAX_POSITION_QUANTITY), Config.FIXED_QUANTITY)
            
            return recommended_size

        except Exception as e:
            logger.error(f"Error calculating recommended position size: {e}")
            return base_size