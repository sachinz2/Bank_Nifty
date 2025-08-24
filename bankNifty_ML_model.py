from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import pytz
from datetime import datetime
import joblib
from pathlib import Path

ist = pytz.timezone('Asia/Kolkata')

class EnhancedMLModel:
    def __init__(self):
        self.model_up = None
        self.model_down = None
        self.feature_list = None
        self.last_training_time = None
        self.performance_metrics = {}
        self.confidence_threshold = Config.ML_BUY_THRESHOLD
        self.models_dir = Path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(self, historical_data, options_data, market_conditions, sentiment_data):
        """Train the enhanced ML model"""
        try:
            logger.info("Training enhanced ML model...")
            
            # Create comprehensive feature set
            features = self._create_feature_matrix(
                historical_data, options_data, market_conditions, sentiment_data
            )
            
            # Create target variables (looking ahead 3 periods for trend)
            future_returns = historical_data['close'].pct_change(3).shift(-3)
            threshold = historical_data['ATR'].rolling(14).mean() * 0.5
            
            y_up = (future_returns > threshold).astype(int)
            y_down = (future_returns < -threshold).astype(int)
            
            # Remove rows with NaN targets
            valid_idx = ~(y_up.isna() | y_down.isna())
            X = features[valid_idx]
            y_up = y_up[valid_idx]
            y_down = y_down[valid_idx]

            # Split data
            X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
                X, y_up, y_down, test_size=0.2, shuffle=False
            )

            # Initialize and train models
            self.model_up = self._train_model(X_train, y_up_train, 'up')
            self.model_down = self._train_model(X_train, y_down_train, 'down')

            # Evaluate models
            self._evaluate_models(X_test, y_up_test, y_down_test)

            # Save models and metadata
            self._save_models()
            
            self.feature_list = X.columns.tolist()
            self.last_training_time = datetime.now(ist)
            
            logger.info("Enhanced ML model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error training enhanced ML model: {e}")
            logger.exception("Full traceback:")
            return False

    def predict(self, market_data, options_data, market_conditions, sentiment_data):
        """Generate trading signals using ML model"""
        try:
            # Create feature matrix for current market state
            current_features = self._create_feature_matrix(
                market_data, options_data, market_conditions, sentiment_data
            )

            # Get model predictions
            up_prob = self.model_up.predict_proba(current_features)[:, 1][-1]
            down_prob = self.model_down.predict_proba(current_features)[:, 1][-1]

            # Calculate dynamic thresholds based on market conditions
            current_threshold = self._calculate_dynamic_threshold(
                market_conditions['volatility'], 
                market_conditions['trend_strength']
            )

            # Generate trading signals
            prediction = {
                'buy_probability': up_prob,
                'sell_probability': down_prob,
                'threshold': current_threshold,
                'confidence': max(up_prob, down_prob),
                'primary_signal': self._determine_primary_signal(up_prob, down_prob, current_threshold),
                'market_state': self._analyze_market_state(market_conditions),
                'timestamp': datetime.now(ist)
            }

            logger.info(f"ML Prediction: {prediction}")
            return prediction

        except Exception as e:
            logger.error(f"Error generating ML prediction: {e}")
            return None

    def _create_feature_matrix(self, market_data, options_data, market_conditions, sentiment_data):
        """Create comprehensive feature matrix for ML model"""
        try:
            features = pd.DataFrame()

            # Technical Features
            features = self._add_technical_features(features, market_data)
            
            # Market Microstructure Features
            features = self._add_microstructure_features(features, market_data)
            
            # Options Features
            features = self._add_options_features(features, options_data)
            
            # Market Condition Features
            features = self._add_market_condition_features(features, market_conditions)
            
            # Sentiment Features
            features = self._add_sentiment_features(features, sentiment_data)
            
            # Normalize features
            features = self._normalize_features(features)
            
            return features

        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return pd.DataFrame()

    def _train_model(self, X_train, y_train, model_type):
        """Train individual XGBoost model"""
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        return model

    def _evaluate_models(self, X_test, y_up_test, y_down_test):
        """Evaluate model performance"""
        try:
            # Get predictions
            up_pred = self.model_up.predict(X_test)
            down_pred = self.model_down.predict(X_test)

            # Calculate metrics
            self.performance_metrics = {
                'up_precision': precision_score(y_up_test, up_pred),
                'up_recall': recall_score(y_up_test, up_pred),
                'up_f1': f1_score(y_up_test, up_pred),
                'down_precision': precision_score(y_down_test, down_pred),
                'down_recall': recall_score(y_down_test, down_pred),
                'down_f1': f1_score(y_down_test, down_pred)
            }

            logger.info(f"Model Performance Metrics: {self.performance_metrics}")

        except Exception as e:
            logger.error(f"Error evaluating models: {e}")

    def _calculate_dynamic_threshold(self, volatility, trend_strength):
        """Calculate dynamic threshold based on market conditions"""
        base_threshold = self.confidence_threshold
        
        # Adjust for volatility
        if volatility == 'high':
            base_threshold *= 1.2
        elif volatility == 'low':
            base_threshold *= 0.9

        # Adjust for trend strength
        if trend_strength > 0.7:
            base_threshold *= 0.9
        elif trend_strength < 0.3:
            base_threshold *= 1.1

        return base_threshold

    def _determine_primary_signal(self, up_prob, down_prob, threshold):
        """Determine primary trading signal"""
        if up_prob > threshold and up_prob > down_prob:
            return 'buy'
        elif down_prob > threshold and down_prob > up_prob:
            return 'sell'
        return 'neutral'

    def _analyze_market_state(self, market_conditions):
        """Analyze current market state for ML context"""
        return {
            'trend': market_conditions.get('trend', 'unknown'),
            'volatility': market_conditions.get('volatility', 'normal'),
            'strength': market_conditions.get('strength', 0),
            'risk_level': self._calculate_risk_level(market_conditions)
        }

    def _calculate_risk_level(self, market_conditions):
        """Calculate current market risk level"""
        risk_score = 0
        
        if market_conditions.get('volatility') == 'high':
            risk_score += 2
        if market_conditions.get('trend') in ['strong_downtrend', 'strong_uptrend']:
            risk_score += 1
        if market_conditions.get('volume_trend') == 'high':
            risk_score += 1
            
        return 'high' if risk_score >= 3 else 'moderate' if risk_score >= 1 else 'low'

    def _save_models(self):
        """Save trained models and metadata"""
        try:
            # Save models
            joblib.dump(self.model_up, self.models_dir / 'enhanced_model_up.joblib')
            joblib.dump(self.model_down, self.models_dir / 'enhanced_model_down.joblib')
            
            # Save performance metrics
            with open(self.models_dir / 'model_metrics.json', 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
                
            logger.info("Models and metrics saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load saved models"""
        try:
            self.model_up = joblib.load(self.models_dir / 'enhanced_model_up.joblib')
            self.model_down = joblib.load(self.models_dir / 'enhanced_model_down.joblib')
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
            
    def _add_technical_features(self, features, market_data):
        """Add technical analysis features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=market_data.index[-1:])
            
            # Price Action Features
            df['close_sma20_ratio'] = market_data['close'] / market_data['SMA20']
            df['close_sma50_ratio'] = market_data['close'] / market_data['SMA50']
            df['close_vwap_ratio'] = market_data['close'] / market_data['VWAP']
            
            # Technical Indicators
            df['rsi'] = market_data['RSI']
            df['macd'] = market_data['MACD']
            df['macd_signal'] = market_data['MACD_Signal']
            df['adx'] = market_data['ADX']
            df['atr_ratio'] = market_data['ATR'] / market_data['close']
            
            # Momentum
            df['roc'] = market_data['ROC']
            df['mfi'] = market_data['MFI']
            
            # Volatility
            df['bb_width'] = (market_data['BB_High'] - market_data['BB_Low']) / market_data['BB_Mid']
            
            # Trend Features
            df['trend_strength'] = np.where(market_data['ADX'] > Config.ADX_THRESHOLD, 1, 0)
            df['above_sma20'] = np.where(market_data['close'] > market_data['SMA20'], 1, 0)
            df['above_sma50'] = np.where(market_data['close'] > market_data['SMA50'], 1, 0)
            
            return df.iloc[-1:] if len(df) > 1 else df

        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return features

    def _add_microstructure_features(self, features, market_data):
        """Add market microstructure features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=market_data.index[-1:])
            
            # Volume Features
            df['volume_sma_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
            df['volume_trend'] = market_data['volume'].pct_change()
            df['obv_trend'] = market_data['OBV'].pct_change()
            
            # Price Impact
            df['high_low_range'] = (market_data['high'] - market_data['low']) / market_data['close']
            df['price_momentum'] = market_data['close'].pct_change()
            
            # Time-based Features
            current_time = pd.Timestamp.now(tz=ist)
            df['time_to_close'] = (pd.Timestamp.combine(current_time.date(), Config.MARKET_CLOSE_TIME) - current_time).seconds / 3600
            df['session_progress'] = (current_time.hour * 60 + current_time.minute) / (6 * 60)  # 6 hours trading session
            
            return df.iloc[-1:] if len(df) > 1 else df

        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
            return features

    def _add_options_features(self, features, options_data):
        """Add options market features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=[0])
            
            # Options Sentiment
            df['put_call_ratio'] = options_data.get('put_call_ratio', 1.0)
            df['iv_skew'] = options_data.get('iv_skew', 0)
            df['atm_iv'] = options_data.get('atm_iv', 0)
            
            # Options Chain Analysis
            chain_analysis = options_data.get('chain_analysis', {})
            df['call_oi_change'] = chain_analysis.get('call_oi_change', 0)
            df['put_oi_change'] = chain_analysis.get('put_oi_change', 0)
            df['max_pain_distance'] = chain_analysis.get('max_pain_distance', 0)
            
            return df

        except Exception as e:
            logger.error(f"Error adding options features: {e}")
            return features

    def _add_market_condition_features(self, features, market_conditions):
        """Add market condition features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=[0])
            
            # Market Trend
            trend_map = {
                'strong_uptrend': 2,
                'uptrend': 1,
                'consolidation': 0,
                'downtrend': -1,
                'strong_downtrend': -2
            }
            df['trend'] = trend_map.get(market_conditions.get('trend', 'consolidation'), 0)
            
            # Market Strength
            df['market_strength'] = market_conditions.get('strength', 50) / 100
            
            # Volatility State
            volatility_map = {'high': 2, 'normal': 1, 'low': 0}
            df['volatility_state'] = volatility_map.get(market_conditions.get('volatility', 'normal'), 1)
            
            # Trading Conditions
            conditions_map = {'favorable': 2, 'moderate': 1, 'high_risk': 0}
            df['trading_conditions'] = conditions_map.get(market_conditions.get('trading_conditions', 'moderate'), 1)
            
            return df

        except Exception as e:
            logger.error(f"Error adding market condition features: {e}")
            return features

    def _add_sentiment_features(self, features, sentiment_data):
        """Add market sentiment features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=[0])
            
            # Market Sentiment
            sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            df['market_sentiment'] = sentiment_map.get(sentiment_data.get('market_sentiment', 'neutral'), 0)
            
            # News and Social Sentiment
            df['news_sentiment'] = sentiment_data.get('news_sentiment', 0)
            df['social_sentiment'] = sentiment_data.get('social_sentiment', 0)
            
            # Fear & Greed
            df['fear_greed_index'] = sentiment_data.get('fear_greed_index', 50) / 100
            
            return df

        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            return features

    def _normalize_features(self, features):
        """Normalize all features"""
        try:
            df = features.copy()
            
            for column in df.columns:
                if df[column].std() != 0:
                    df[column] = (df[column] - df[column].mean()) / df[column].std()
                else:
                    df[column] = 0
            
            return df

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features

    def get_feature_importance(self):
        """Get feature importance from both models"""
        try:
            if self.model_up is None or self.model_down is None:
                return None

            importance_up = pd.DataFrame({
                'feature': self.feature_list,
                'importance_up': self.model_up.feature_importances_
            })
            
            importance_down = pd.DataFrame({
                'feature': self.feature_list,
                'importance_down': self.model_down.feature_importances_
            })
            
            importance = importance_up.merge(importance_down, on='feature')
            importance['total_importance'] = importance['importance_up'] + importance['importance_down']
            
            return importance.sort_values('total_importance', ascending=False)

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None

    def should_retrain(self):
        """Check if model should be retrained"""
        try:
            if self.last_training_time is None:
                return True
                
            hours_since_training = (datetime.now(ist) - self.last_training_time).total_seconds() / 3600
            
            # Check time-based criteria
            if hours_since_training >= Config.ML_RETRAIN_HOURS:
                return True
                
            # Check performance-based criteria
            if self.performance_metrics:
                if (self.performance_metrics.get('up_f1', 0) < 0.6 or 
                    self.performance_metrics.get('down_f1', 0) < 0.6):
                    return True
            
            return False

        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")
            return False