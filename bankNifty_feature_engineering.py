import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import pytz
from scipy.stats import norm
from datetime import datetime, timedelta

ist = pytz.timezone('Asia/Kolkata')

class FeatureEngineer:
    def __init__(self):
        self.feature_cache = {}
        self.last_update = None
        self.required_features = self._get_required_features()

    def _get_required_features(self):
        """Define all required features for the ML model"""
        return {
            'price_features': [
                'sma20_ratio', 'sma50_ratio', 'ema20_ratio',
                'price_momentum', 'returns_5m', 'returns_15m', 'returns_1h'
            ],
            'technical_features': [
                'rsi', 'macd', 'macd_signal', 'bb_position', 'atr_ratio',
                'obv_trend', 'mfi', 'cci', 'adx', 'aroon_oscillator'
            ],
            'volatility_features': [
                'historical_volatility', 'implied_volatility_ratio',
                'vix_ratio', 'atr_volatility'
            ],
            'volume_features': [
                'volume_sma_ratio', 'volume_momentum', 'volume_trend',
                'obv_sma_ratio', 'volume_intensity'
            ],
            'options_features': [
                'put_call_ratio', 'iv_skew', 'max_pain_distance',
                'option_momentum', 'delta_neutral_ratio'
            ],
            'market_features': [
                'market_breadth', 'advance_decline_ratio',
                'sector_performance', 'market_momentum'
            ],
            'sentiment_features': [
                'sentiment_score', 'fear_greed_index',
                'news_sentiment', 'options_sentiment'
            ],
            'time_features': [
                'time_to_expiry', 'time_of_day', 'day_of_week'
            ]
        }

    def engineer_features(self, historical_data, options_data, sentiment_data, 
                         market_breadth_data, volatility_data):
        """Main feature engineering function"""
        try:
            logger.info("Starting feature engineering process...")
            
            if self._should_use_cached_features():
                return self.feature_cache
            
            features = pd.DataFrame(index=[0])
            
            # Add each feature group
            features = features.join(self._create_price_features(historical_data))
            features = features.join(self._create_technical_features(historical_data))
            features = features.join(self._create_volatility_features(historical_data, volatility_data))
            features = features.join(self._create_volume_features(historical_data))
            features = features.join(self._create_options_features(options_data))
            features = features.join(self._create_market_features(market_breadth_data))
            features = features.join(self._create_sentiment_features(sentiment_data))
            features = features.join(self._create_time_features())
            
            # Add interaction features
            features = self._add_interaction_features(features)
            
            # Normalize features
            features = self._normalize_features(features)
            
            # Cache the features
            self._cache_features(features)
            
            logger.info(f"Feature engineering completed. Created {len(features.columns)} features")
            return features

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            logger.exception("Full traceback:")
            return pd.DataFrame()

    def _create_price_features(self, data):
        """Create price-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            close = data['close'].iloc[-1]
            
            # Moving average ratios
            features['sma20_ratio'] = close / data['close'].rolling(20).mean().iloc[-1]
            features['sma50_ratio'] = close / data['close'].rolling(50).mean().iloc[-1]
            features['ema20_ratio'] = close / data['close'].ewm(span=20).mean().iloc[-1]
            
            # Price momentum
            features['price_momentum'] = (close - data['close'].iloc[-5]) / data['close'].iloc[-5]
            
            # Returns over different timeframes
            features['returns_5m'] = data['close'].pct_change(1).iloc[-1]
            features['returns_15m'] = data['close'].pct_change(3).iloc[-1]
            features['returns_1h'] = data['close'].pct_change(12).iloc[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            return pd.DataFrame()

    def _create_technical_features(self, data):
        """Create technical indicator features"""
        try:
            features = pd.DataFrame(index=[0])
            
            # RSI
            features['rsi'] = momentum.RSIIndicator(data['close']).rsi().iloc[-1]
            
            # MACD
            macd_ind = trend.MACD(data['close'])
            features['macd'] = macd_ind.macd().iloc[-1]
            features['macd_signal'] = macd_ind.macd_signal().iloc[-1]
            features['macd_diff'] = macd_ind.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb_ind = volatility.BollingerBands(data['close'])
            features['bb_position'] = (data['close'].iloc[-1] - bb_ind.bollinger_lband().iloc[-1]) / \
                                    (bb_ind.bollinger_hband().iloc[-1] - bb_ind.bollinger_lband().iloc[-1])
            
            # ATR
            features['atr_ratio'] = volatility.AverageTrueRange(
                data['high'], data['low'], data['close']
            ).average_true_range().iloc[-1] / data['close'].iloc[-1]
            
            # Additional technical indicators
            features['cci'] = trend.CCIIndicator(
                data['high'], data['low'], data['close']
            ).cci().iloc[-1]
            
            features['adx'] = trend.ADXIndicator(
                data['high'], data['low'], data['close']
            ).adx().iloc[-1]
            
            aroon_ind = trend.AroonIndicator(data['close'])
            features['aroon_oscillator'] = (
                aroon_ind.aroon_up() - aroon_ind.aroon_down()
            ).iloc[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return pd.DataFrame()

    def _create_volatility_features(self, data, volatility_data):
        """Create volatility-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            # Historical volatility
            returns = data['close'].pct_change()
            features['historical_volatility'] = returns.std() * np.sqrt(252)
            
            # Implied volatility features
            if volatility_data is not None:
                features['iv_ratio'] = volatility_data.get('current_iv', 0) / \
                                     volatility_data.get('historical_iv', 1)
                features['vix_ratio'] = volatility_data.get('vix', 15) / \
                                      volatility_data.get('vix_average', 15)
            
            # ATR-based volatility
            atr = volatility.AverageTrueRange(
                data['high'], data['low'], data['close']
            ).average_true_range()
            features['atr_volatility'] = atr.iloc[-1] / atr.rolling(20).mean().iloc[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating volatility features: {e}")
            return pd.DataFrame()

    def _create_volume_features(self, data):
        """Create volume-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            # Volume ratios
            features['volume_sma_ratio'] = data['volume'].iloc[-1] / \
                                         data['volume'].rolling(20).mean().iloc[-1]
            
            # Volume momentum
            features['volume_momentum'] = (data['volume'].iloc[-1] - data['volume'].iloc[-5]) / \
                                        data['volume'].iloc[-5]
            
            # OBV trend
            obv = volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
            features['obv_trend'] = (obv.iloc[-1] - obv.iloc[-5]) / obv.iloc[-5]
            
            # Volume intensity
            features['volume_intensity'] = data['volume'].iloc[-1] / data['volume'].iloc[-20:].mean()
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
            return pd.DataFrame()

    def _create_options_features(self, options_data):
        """Create options-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            if options_data is not None:
                # Basic options ratios
                features['put_call_ratio'] = options_data.get('put_call_ratio', 1.0)
                features['iv_skew'] = options_data.get('iv_skew', 0.0)
                
                # Options sentiment
                features['options_sentiment'] = 1 if options_data.get('sentiment') == 'bullish' else \
                                              -1 if options_data.get('sentiment') == 'bearish' else 0
                
                # Distance from max pain
                if 'max_pain' in options_data and 'current_price' in options_data:
                    features['max_pain_distance'] = (
                        options_data['current_price'] - options_data['max_pain']
                    ) / options_data['max_pain']
                
                # Option chain analysis
                if 'chain_analysis' in options_data:
                    chain = options_data['chain_analysis']
                    features['call_oi_change'] = chain.get('call_oi_change', 0)
                    features['put_oi_change'] = chain.get('put_oi_change', 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating options features: {e}")
            return pd.DataFrame()

    def _create_market_features(self, market_data):
        """Create market-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            if market_data is not None:
                # Market breadth
                features['advance_decline_ratio'] = market_data.get('advance_decline_ratio', 1.0)
                features['market_breadth'] = market_data.get('market_breadth', 0.0)
                
                # Sector performance
                if 'sector_performance' in market_data:
                    features['sector_rel_strength'] = market_data['sector_performance'].get(
                        'relative_strength', 0.0
                    )
                
                # Market momentum
                features['market_momentum'] = market_data.get('market_momentum', 0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating market features: {e}")
            return pd.DataFrame()

    def _create_sentiment_features(self, sentiment_data):
        """Create sentiment-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            if sentiment_data is not None:
                # Market sentiment
                features['sentiment_score'] = sentiment_data.get('sentiment_score', 0)
                features['fear_greed_index'] = sentiment_data.get('fear_greed_index', 50)
                
                # News sentiment
                features['news_sentiment'] = sentiment_data.get('news_sentiment', 0)
                
                # Social media sentiment
                features['social_sentiment'] = sentiment_data.get('social_sentiment', 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating sentiment features: {e}")
            return pd.DataFrame()

    def _create_time_features(self):
        """Create time-based features"""
        try:
            features = pd.DataFrame(index=[0])
            
            current_time = datetime.now(ist)
            
            # Time of day features
            features['time_of_day'] = current_time.hour + current_time.minute / 60
            features['day_of_week'] = current_time.weekday()
            
            # Market session
            features['is_morning_session'] = 1 if current_time.hour < 12 else 0
            features['is_afternoon_session'] = 1 if 12 <= current_time.hour < 15 else 0
            
            # Time to market events
            market_close = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
            features['time_to_close'] = (market_close - current_time).total_seconds() / 3600
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return pd.DataFrame()

    def _add_interaction_features(self, features):
        """Add interaction features between important indicators"""
        try:
            # RSI and Volume
            if 'rsi' in features.columns and 'volume_sma_ratio' in features.columns:
                features['rsi_volume_interaction'] = features['rsi'] * features['volume_sma_ratio']
            
            # MACD and ADX
            if 'macd' in features.columns and 'adx' in features.columns:
                features['macd_adx_interaction'] = features['macd'] * features['adx']
            
            # Sentiment and Volume
            if 'sentiment_score' in features.columns and 'volume_intensity' in features.columns:
                features['sentiment_volume_interaction'] = \
                    features['sentiment_score'] * features['volume_intensity']
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            return features

    def _normalize_features(self, features):
        """Normalize all features to a similar scale"""
        try:
            for column in features.columns:
                if features[column].std() != 0:
                    features[column] = (features[column] - features[column].mean()) / \
                                     features[column].std()
                else:
                    features[column] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features

    def _should_use_cached_features(self):
        """Determine if cached features can be used"""
        if self.last_update is None:
            return False
        
        time_since_update = (datetime.now(ist) - self.last_update).total_seconds()
        return time_since_update < Config.FEATURE_CACHE_SECONDS

    def _cache_features(self, features):
        """Cache the computed features"""
        try:
            self.feature_cache = features.copy()
            self.last_update = datetime.now(ist)
            logger.info("Features cached successfully")
        except Exception as e:
            logger.error(f"Error caching features: {e}")

    def preprocess_features(self, features):
        """Preprocess features for ML model consumption"""
        try:
            processed_features = features.copy()
            
            # Handle missing values
            processed_features = self._handle_missing_values(processed_features)
            
            # Handle outliers
            processed_features = self._handle_outliers(processed_features)
            
            # Ensure all features are numeric
            processed_features = self._ensure_numeric(processed_features)
            
            # Scale features
            processed_features = self._scale_features(processed_features)
            
            # Validate processed features
            if self._validate_features(processed_features):
                return processed_features
            else:
                logger.error("Feature validation failed")
                return None
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            logger.exception("Full traceback:")
            return None

    def _handle_missing_values(self, features):
        """Handle missing values in features"""
        try:
            # Forward fill for time series data
            features = features.ffill()
            
            # Fill remaining NaN values with appropriate values
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            categorical_columns = features.select_dtypes(exclude=[np.number]).columns
            
            # For numeric columns
            for col in numeric_columns:
                if features[col].isnull().any():
                    if 'ratio' in col or 'percentage' in col:
                        features[col].fillna(1.0, inplace=True)
                    else:
                        features[col].fillna(features[col].mean(), inplace=True)
            
            # For categorical columns
            for col in categorical_columns:
                features[col].fillna(features[col].mode()[0], inplace=True)
            
            return features
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return features

    def _handle_outliers(self, features):
        """Handle outliers in feature data"""
        try:
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                # Calculate IQR
                Q1 = features[column].quantile(0.25)
                Q3 = features[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                features[column] = features[column].clip(lower_bound, upper_bound)
            
            return features
            
        except Exception as e:
            logger.error(f"Error handling outliers: {e}")
            return features

    def _ensure_numeric(self, features):
        """Ensure all features are numeric"""
        try:
            # Convert boolean columns
            bool_columns = features.select_dtypes(include=[bool]).columns
            for col in bool_columns:
                features[col] = features[col].astype(int)
            
            # Convert categorical columns
            cat_columns = features.select_dtypes(include=['object', 'category']).columns
            for col in cat_columns:
                features[col] = pd.Categorical(features[col]).codes
            
            # Verify all columns are numeric
            non_numeric = features.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                logger.warning(f"Non-numeric columns found: {non_numeric}")
                features = features.select_dtypes(include=[np.number])
            
            return features
            
        except Exception as e:
            logger.error(f"Error ensuring numeric features: {e}")
            return features

    def _scale_features(self, features):
        """Scale features to appropriate ranges"""
        try:
            for column in features.columns:
                # Determine scaling method based on feature type
                if 'ratio' in column or 'percentage' in column:
                    # Min-max scaling for ratio features
                    min_val = features[column].min()
                    max_val = features[column].max()
                    if max_val > min_val:
                        features[column] = (features[column] - min_val) / (max_val - min_val)
                else:
                    # Z-score scaling for other features
                    mean_val = features[column].mean()
                    std_val = features[column].std()
                    if std_val > 0:
                        features[column] = (features[column] - mean_val) / std_val
            
            return features
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return features

    def _validate_features(self, features):
        """Validate processed features"""
        try:
            # Check for required features
            missing_features = set(self.required_features.values()) - set(features.columns)
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return False
            
            # Check for invalid values
            if features.isnull().any().any():
                logger.error("Features contain null values after processing")
                return False
            
            if np.isinf(features.values).any():
                logger.error("Features contain infinite values")
                return False
            
            # Check feature ranges
            if (features.min() < -10).any() or (features.max() > 10).any():
                logger.warning("Features contain values outside expected range")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return False

    def get_feature_importance(self, model):
        """Get feature importance from the ML model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_cache.columns,
                    'importance': importance
                })
                feature_imp = feature_imp.sort_values('importance', ascending=False)
                return feature_imp
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None

    def save_feature_metadata(self):
        """Save feature engineering metadata"""
        try:
            metadata = {
                'feature_count': len(self.feature_cache.columns),
                'feature_list': self.feature_cache.columns.tolist(),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'feature_stats': {
                    col: {
                        'mean': float(self.feature_cache[col].mean()),
                        'std': float(self.feature_cache[col].std()),
                        'min': float(self.feature_cache[col].min()),
                        'max': float(self.feature_cache[col].max())
                    }
                    for col in self.feature_cache.columns
                }
            }
            
            with open(Config.DATA_DIR / 'feature_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Feature metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving feature metadata: {e}")

if __name__ == "__main__":
    # Testing feature engineering
    try:
        engineer = FeatureEngineer()
        # Add test code here
        pass
    except Exception as e:
        logger.error(f"Error in feature engineering test: {e}")