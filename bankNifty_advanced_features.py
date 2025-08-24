import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from ta import momentum, trend, volatility, volume
from logging_utils import banknifty_logger as logger
from typing import Dict, List, Tuple, Any
import talib
from bankNifty_config import Config

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_importance = {}
        self.selected_features = set()
        self.pca_components = None
        self.feature_scaler = None

    def engineer_features(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Engineer comprehensive feature set"""
        try:
            features = pd.DataFrame()
            
            # Basic technical features
            features = self._add_technical_features(features, market_data['historical_data'])
            
            # Advanced pattern features
            features = self._add_pattern_features(features, market_data['historical_data'])
            
            # Market microstructure features
            features = self._add_microstructure_features(features, market_data)
            
            # Options-based features
            features = self._add_options_features(features, market_data['options_data'])
            
            # Sentiment and market breadth features
            features = self._add_market_context_features(features, market_data)
            
            # Time-based features
            features = self._add_time_features(features)
            
            # Interaction features
            features = self._add_interaction_features(features)
            
            # Feature selection and dimensionality reduction
            features = self._optimize_feature_set(features)
            
            return features

        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            logger.exception("Full traceback:")
            return pd.DataFrame()

    def _add_technical_features(self, features: pd.DataFrame, 
                              historical_data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=historical_data.index[-1:])
            
            # Standard technical indicators
            df = self._add_trend_features(df, historical_data)
            df = self._add_momentum_features(df, historical_data)
            df = self._add_volatility_features(df, historical_data)
            df = self._add_volume_features(df, historical_data)
            
            # Custom technical features
            df = self._add_custom_indicators(df, historical_data)
            
            return df

        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return features

    def _add_pattern_features(self, features: pd.DataFrame, 
                            historical_data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick and chart pattern features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=historical_data.index[-1:])
            
            # Candlestick patterns using talib
            patterns = {
                'CDL_HAMMER': talib.CDLHAMMER,
                'CDL_ENGULFING': talib.CDLENGULFING,
                'CDL_DOJI': talib.CDLDOJI,
                'CDL_EVENINGSTAR': talib.CDLEVENINGSTAR,
                'CDL_MORNINGSTAR': talib.CDLMORNINGSTAR
            }
            
            for pattern_name, pattern_func in patterns.items():
                df[f'pattern_{pattern_name}'] = pattern_func(
                    historical_data['open'].values,
                    historical_data['high'].values,
                    historical_data['low'].values,
                    historical_data['close'].values
                )
            
            # Chart patterns
            df = self._add_chart_patterns(df, historical_data)
            
            return df

        except Exception as e:
            logger.error(f"Error adding pattern features: {e}")
            return features

    def _add_microstructure_features(self, features: pd.DataFrame, 
                                   market_data: Dict[str, Any]) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=[0])
            historical_data = market_data['historical_data']
            
            # Trade size analysis
            df['avg_trade_size'] = historical_data['volume'] * historical_data['close'] / \
                                 historical_data['volume'].rolling(20).count()
            
            # Price impact
            df['price_impact'] = (historical_data['high'] - historical_data['low']) / \
                               historical_data['volume']
            
            # Bid-ask bounce
            df['tick_reversal'] = (
                (historical_data['close'] - historical_data['open']).diff().apply(np.sign) * -1
            ).rolling(5).sum()
            
            # Volume pressure
            df['volume_pressure'] = historical_data['volume'].diff() / \
                                  historical_data['volume'].rolling(5).std()
            
            # Price efficiency
            df['price_efficiency'] = self._calculate_price_efficiency(historical_data)
            
            return df

        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
            return features

    def _add_options_features(self, features: pd.DataFrame, 
                            options_data: Dict[str, Any]) -> pd.DataFrame:
        """Add advanced options-based features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=[0])
            
            if not options_data:
                return df
                
            # Basic options ratios
            df['put_call_ratio'] = options_data.get('put_call_ratio', 1.0)
            df['iv_skew'] = options_data.get('iv_skew', 0.0)
            
            # Option chain analysis
            chain = options_data.get('chain_analysis', {})
            
            # Volume concentration
            df['call_volume_concentration'] = self._calculate_volume_concentration(
                chain.get('call_strikes', []),
                chain.get('call_volumes', [])
            )
            df['put_volume_concentration'] = self._calculate_volume_concentration(
                chain.get('put_strikes', []),
                chain.get('put_volumes', [])
            )
            
            # Strike distance metrics
            df['strike_distance_weighted'] = self._calculate_strike_distance(
                chain.get('current_price', 0),
                chain.get('call_strikes', []),
                chain.get('put_strikes', [])
            )
            
            return df

        except Exception as e:
            logger.error(f"Error adding options features: {e}")
            return features

    def _add_market_context_features(self, features: pd.DataFrame, 
                                   market_data: Dict[str, Any]) -> pd.DataFrame:
        """Add market context and sentiment features"""
        try:
            df = features.copy() if not features.empty else pd.DataFrame(index=[0])
            
            # Market breadth
            breadth_data = market_data.get('market_breadth', {})
            df['market_breadth'] = breadth_data.get('advance_decline_ratio', 1.0)
            df['breadth_momentum'] = self._calculate_breadth_momentum(breadth_data)
            
            # Sentiment metrics
            sentiment_data = market_data.get('sentiment_data', {})
            df['sentiment_score'] = sentiment_data.get('sentiment_score', 0)
            df['sentiment_change'] = sentiment_data.get('sentiment_change', 0)
            
            # Market condition features
            df['market_regime'] = self._identify_market_regime(market_data['historical_data'])
            df['trend_strength'] = self._calculate_trend_strength(market_data['historical_data'])
            
            # Volatility regime
            df['volatility_regime'] = self._identify_volatility_regime(market_data['historical_data'])
            
            # Correlation features
            df['sector_correlation'] = self._calculate_sector_correlation(market_data)
            
            return df

    def _identify_market_regime(self, historical_data: pd.DataFrame) -> float:
        """Identify current market regime using multiple indicators"""
        try:
            # Calculate regime indicators
            sma_ratio = historical_data['close'] / historical_data['SMA50']
            volatility = historical_data['close'].pct_change().rolling(20).std()
            trend_direction = self._calculate_trend_direction(historical_data)
            
            # Combine indicators into regime score
            regime_score = 0
            
            # Trend component
            regime_score += np.sign(sma_ratio.iloc[-1] - 1) * abs(sma_ratio.iloc[-1] - 1)
            
            # Volatility component
            vol_zscore = (volatility.iloc[-1] - volatility.mean()) / volatility.std()
            regime_score += np.sign(vol_zscore) * min(abs(vol_zscore), 3) / 3
            
            # Trend strength component
            regime_score += trend_direction * 0.5
            
            return regime_score
            
        except Exception as e:
            logger.error(f"Error identifying market regime: {e}")
            return 0.0

    def _calculate_trend_strength(self, historical_data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple methods"""
        try:
            # ADX-based trend strength
            adx = historical_data['ADX'].iloc[-1] / 100.0
            
            # Price momentum
            momentum = (historical_data['close'].iloc[-1] / historical_data['close'].iloc[-20] - 1)
            momentum_score = min(abs(momentum), 0.1) / 0.1  # Normalize to [0, 1]
            
            # Linear regression R² of price
            prices = historical_data['close'].iloc[-20:]
            x = np.arange(len(prices))
            z = np.polyfit(x, prices, 1)
            p = np.poly1d(z)
            r2 = 1 - np.sum((prices - p(x))**2) / np.sum((prices - prices.mean())**2)
            
            # Combine metrics
            trend_strength = (adx * 0.4 + momentum_score * 0.3 + r2 * 0.3)
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0

    def _identify_volatility_regime(self, historical_data: pd.DataFrame) -> int:
        """Identify volatility regime using multiple timeframes"""
        try:
            # Calculate volatility at different timeframes
            vol_5 = historical_data['close'].pct_change().rolling(5).std() * np.sqrt(252)
            vol_20 = historical_data['close'].pct_change().rolling(20).std() * np.sqrt(252)
            vol_60 = historical_data['close'].pct_change().rolling(60).std() * np.sqrt(252)
            
            # Get current volatility levels
            current_vol = {
                'short': vol_5.iloc[-1],
                'medium': vol_20.iloc[-1],
                'long': vol_60.iloc[-1]
            }
            
            # Determine regime
            if all(v > Config.HIGH_VOLATILITY_THRESHOLD for v in current_vol.values()):
                return 2  # High volatility regime
            elif all(v < Config.LOW_VOLATILITY_THRESHOLD for v in current_vol.values()):
                return 0  # Low volatility regime
            return 1  # Normal volatility regime
            
        except Exception as e:
            logger.error(f"Error identifying volatility regime: {e}")
            return 1

    def _calculate_sector_correlation(self, market_data: Dict[str, Any]) -> float:
        """Calculate correlation with sector movement"""
        try:
            if 'sector_data' not in market_data:
                return 0.0
                
            sector_returns = market_data['sector_data']['returns']
            stock_returns = market_data['historical_data']['close'].pct_change()
            
            correlation = stock_returns.corr(sector_returns)
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating sector correlation: {e}")
            return 0.0

    def _optimize_feature_set(self, features: pd.DataFrame) -> pd.DataFrame:
        """Optimize feature set using multiple techniques"""
        try:
            if features.empty:
                return features

            # Remove highly correlated features
            features = self._remove_correlated_features(features)
            
            # Feature selection using mutual information
            if len(features.columns) > Config.MAX_FEATURES:
                features = self._select_best_features(features)
            
            # Apply PCA if specified
            if Config.USE_PCA:
                features = self._apply_pca(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error optimizing feature set: {e}")
            return features

    def _remove_correlated_features(self, features: pd.DataFrame, 
                                  threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find highly correlated feature pairs
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper.columns 
                      if any(upper[column] > threshold)]
            
            # Remove features
            features_reduced = features.drop(to_drop, axis=1)
            
            logger.info(f"Removed {len(to_drop)} highly correlated features")
            return features_reduced
            
        except Exception as e:
            logger.error(f"Error removing correlated features: {e}")
            return features

    def _select_best_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select best features using mutual information"""
        try:
            # Generate target variable (can be modified based on strategy)
            target = (features['close'].shift(-1) > features['close']).astype(int)
            
            # Remove target from features
            features_for_selection = features.drop('close', axis=1)
            
            # Select best features
            selector = SelectKBest(score_func=mutual_info_classif, 
                                 k=min(Config.MAX_FEATURES, len(features_for_selection.columns)))
            
            selected_features = selector.fit_transform(features_for_selection, target)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_names = features_for_selection.columns[selected_mask].tolist()
            
            # Keep track of selected features
            self.selected_features = set(selected_names)
            
            # Return selected features plus target
            return features[['close'] + selected_names]
            
        except Exception as e:
            logger.error(f"Error selecting best features: {e}")
            return features

    def _apply_pca(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        try:
            if self.pca_components is None:
                # Initialize PCA
                pca = PCA(n_components=0.95)  # Keep 95% of variance
                self.pca_components = pca.fit(features)
            
            # Transform features
            transformed_features = self.pca_components.transform(features)
            
            # Convert back to DataFrame
            pca_df = pd.DataFrame(
                transformed_features,
                index=features.index,
                columns=[f'PC_{i+1}' for i in range(transformed_features.shape[1])]
            )
            
            return pca_df
            
        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            return features

    def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance rankings"""
        return self.feature_importance

    def update_feature_importance(self, model_importance: Dict[str, float]):
        """Update feature importance based on model feedback"""
        try:
            # Exponential moving average of feature importance
            alpha = 0.7
            for feature, importance in model_importance.items():
                current_importance = self.feature_importance.get(feature, importance)
                self.feature_importance[feature] = alpha * importance + (1 - alpha) * current_importance
                
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")

# Additional helper methods can be added here for specific feature calculations

if __name__ == "__main__":
    # Add test code here
    pass