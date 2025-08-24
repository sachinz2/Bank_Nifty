import pandas as pd
import numpy as np
from logging_utils import banknifty_logger as logger
from typing import Dict, Any, Optional, Union
import json
from pathlib import Path

class DataValidator:
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.error_records = []
        self.recovery_methods = {
            'missing_columns': self._handle_missing_columns,
            'invalid_values': self._handle_invalid_values,
            'data_type_mismatch': self._handle_data_type_mismatch,
            'missing_timeframes': self._handle_missing_timeframes
        }

    def validate_market_data(self, 
                           data: Dict[str, Any], 
                           raise_errors: bool = False) -> Dict[str, Any]:
        """Validate and clean market data"""
        try:
            validation_results = {
                'historical_data': self._validate_historical_data(data.get('historical_data')),
                'options_data': self._validate_options_data(data.get('options_data')),
                'sentiment_data': self._validate_sentiment_data(data.get('sentiment_data')),
                'market_breadth': self._validate_market_breadth(data.get('market_breadth')),
                'technical_data': self._validate_technical_data(data.get('technical_data'))
            }

            if not all(validation_results.values()):
                error_msg = f"Data validation failed: {validation_results}"
                if raise_errors:
                    raise ValueError(error_msg)
                logger.error(error_msg)

            # Clean and recover data where possible
            cleaned_data = self._clean_data(data, validation_results)
            
            return cleaned_data

        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            logger.exception("Full traceback:")
            if raise_errors:
                raise
            return self._get_safe_default_data()

    def _validate_historical_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate historical price data"""
        try:
            if df is None or df.empty:
                return {'valid': False, 'reason': 'empty_data'}

            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not all(col in df.columns for col in required_columns):
                return {'valid': False, 'reason': 'missing_columns'}

            # Check for invalid values
            for col in required_columns:
                if df[col].isnull().any() or (df[col] <= 0).any():
                    return {'valid': False, 'reason': 'invalid_values'}

            # Check time series continuity
            if not self._check_time_continuity(df):
                return {'valid': False, 'reason': 'time_discontinuity'}

            return {'valid': True, 'reason': None}

        except Exception as e:
            logger.error(f"Error validating historical data: {e}")
            return {'valid': False, 'reason': 'validation_error'}

    def _validate_options_data(self, options_data: Dict) -> Dict[str, bool]:
        """Validate options data"""
        try:
            if not options_data:
                return {'valid': False, 'reason': 'empty_data'}

            required_fields = {
                'put_call_ratio', 'iv_skew', 'chain_analysis'
            }
            
            if not all(field in options_data for field in required_fields):
                return {'valid': False, 'reason': 'missing_fields'}

            # Validate chain analysis
            chain_analysis = options_data.get('chain_analysis', {})
            if not self._validate_chain_analysis(chain_analysis):
                return {'valid': False, 'reason': 'invalid_chain_analysis'}

            return {'valid': True, 'reason': None}

        except Exception as e:
            logger.error(f"Error validating options data: {e}")
            return {'valid': False, 'reason': 'validation_error'}

    def _clean_data(self, data: Dict[str, Any], 
                    validation_results: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
        """Clean and recover data where possible"""
        try:
            cleaned_data = {}
            
            for key, validation in validation_results.items():
                if not validation['valid']:
                    recovery_method = self.recovery_methods.get(validation['reason'])
                    if recovery_method:
                        cleaned_data[key] = recovery_method(data.get(key))
                    else:
                        cleaned_data[key] = self._get_safe_default_data_for_type(key)
                else:
                    cleaned_data[key] = data.get(key)

            return cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return self._get_safe_default_data()

    def _handle_missing_columns(self, data: Union[pd.DataFrame, Dict]) -> Union[pd.DataFrame, Dict]:
        """Handle missing columns in data"""
        try:
            if isinstance(data, pd.DataFrame):
                required_columns = self.validation_rules['required_columns']
                for col in required_columns:
                    if col not in data.columns:
                        # Add column with safe default values
                        data[col] = self._get_safe_default_value(col)
            return data
        except Exception as e:
            logger.error(f"Error handling missing columns: {e}")
            return pd.DataFrame()

    def _handle_invalid_values(self, data: Union[pd.DataFrame, Dict]) -> Union[pd.DataFrame, Dict]:
        """Handle invalid values in data"""
        try:
            if isinstance(data, pd.DataFrame):
                # Replace invalid values with forward fill or safe defaults
                for col in data.columns:
                    mask = (data[col].isnull()) | (data[col] <= 0)
                    if mask.any():
                        data.loc[mask, col] = data[col].ffill().fillna(self._get_safe_default_value(col))
            return data
        except Exception as e:
            logger.error(f"Error handling invalid values: {e}")
            return pd.DataFrame()

    def _check_time_continuity(self, df: pd.DataFrame) -> bool:
        """Check for time series continuity"""
        try:
            if 'timestamp' not in df.columns:
                return True  # Can't check continuity without timestamp
                
            time_diff = df['timestamp'].diff()
            expected_diff = pd.Timedelta(minutes=5)  # Assuming 5-minute data
            
            return not (time_diff != expected_diff).any()
        except Exception as e:
            logger.error(f"Error checking time continuity: {e}")
            return False

    def _get_safe_default_value(self, column: str) -> Union[float, int, str]:
        """Get safe default value for a column"""
        default_values = {
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0,
            'volume': 0,
            'vwap': 0.0,
            'rsi': 50.0,
            'macd': 0.0,
            'sentiment': 'neutral'
        }
        return default_values.get(column, 0.0)

    def _validate_chain_analysis(self, chain_analysis: Dict) -> bool:
        """Validate options chain analysis data"""
        required_fields = {
            'call_oi_change', 'put_oi_change', 'max_pain'
        }
        return all(field in chain_analysis for field in required_fields)

    def _get_safe_default_data(self) -> Dict[str, Any]:
        """Get safe default data structure"""
        return {
            'historical_data': pd.DataFrame(),
            'options_data': {'put_call_ratio': 1.0, 'iv_skew': 0.0, 'chain_analysis': {}},
            'sentiment_data': {'market_sentiment': 'neutral', 'strength': 0},
            'market_breadth': {'advance_decline_ratio': 1.0, 'market_breadth': 0},
            'technical_data': pd.DataFrame()
        }

    @staticmethod
    def _load_validation_rules():
        """Load validation rules from configuration"""
        return {
            'required_columns': {
                'historical_data': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'technical_data': [
                    'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_Signal', 
                    'BB_High', 'BB_Low', 'ATR', 'ADX'
                ]
            },
            'value_ranges': {
                'RSI': (0, 100),
                'ADX': (0, 100),
                'volume': (0, float('inf')),
                'ATR': (0, float('inf'))
            },
            'time_series_checks': {
                'max_gaps_allowed': 3,
                'min_data_points': 100
            }
        }

    def add_error_record(self, error_type: str, details: Dict[str, Any]):
        """Record validation errors for analysis"""
        self.error_records.append({
            'timestamp': pd.Timestamp.now(tz='Asia/Kolkata'),
            'error_type': error_type,
            'details': details
        })
        
        # Save error records periodically
        if len(self.error_records) >= 100:
            self._save_error_records()

    def _save_error_records(self):
        """Save error records to file"""
        try:
            error_file = Path(Config.DATA_DIR) / 'validation_errors.json'
            with open(error_file, 'w') as f:
                json.dump(self.error_records, f, indent=2, default=str)
            self.error_records = []
        except Exception as e:
            logger.error(f"Error saving validation records: {e}")

    def validate_feature_data(self, features: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """Validate engineered features"""
        try:
            if features is None or features.empty:
                return False, pd.DataFrame()

            # Check for required features
            missing_features = self._check_missing_features(features)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                features = self._handle_missing_features(features, missing_features)

            # Check for invalid values
            features = self._handle_invalid_feature_values(features)

            # Check for proper scaling
            features = self._validate_feature_scaling(features)

            return True, features

        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return False, pd.DataFrame()

    def _check_missing_features(self, features: pd.DataFrame) -> List[str]:
        """Check for missing required features"""
        required_features = {
            # Price features
            'close_sma20_ratio', 'close_sma50_ratio', 'price_momentum',
            # Technical features
            'rsi', 'macd', 'adx', 'atr_ratio',
            # Volume features
            'volume_sma_ratio', 'obv_trend',
            # Options features
            'put_call_ratio', 'iv_skew'
        }
        return list(required_features - set(features.columns))

    def _handle_missing_features(self, features: pd.DataFrame, 
                               missing_features: List[str]) -> pd.DataFrame:
        """Handle missing features with safe defaults"""
        for feature in missing_features:
            if 'ratio' in feature:
                features[feature] = 1.0
            elif feature in ['rsi', 'adx']:
                features[feature] = 50.0
            elif feature in ['macd', 'atr_ratio', 'momentum']:
                features[feature] = 0.0
            else:
                features[feature] = 0.0
        return features

    def _handle_invalid_feature_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle invalid feature values"""
        # Replace infinities
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values
        for col in features.columns:
            if features[col].isna().any():
                if 'ratio' in col:
                    features[col] = features[col].fillna(1.0)
                else:
                    features[col] = features[col].fillna(features[col].mean())
        
        return features

    def _validate_feature_scaling(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix feature scaling"""
        for col in features.columns:
            # Check for extreme values
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                z_scores = (features[col] - mean) / std
                outliers = (z_scores.abs() > 3)
                if outliers.any():
                    # Cap outliers at 3 standard deviations
                    features.loc[outliers, col] = np.sign(z_scores[outliers]) * 3 * std + mean
        
        return features

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        return {
            'error_summary': self._summarize_errors(),
            'data_quality_metrics': self._calculate_quality_metrics(),
            'validation_recommendations': self._generate_recommendations(),
            'timestamp': pd.Timestamp.now(tz='Asia/Kolkata')
        }

    def _summarize_errors(self) -> Dict[str, Any]:
        """Summarize validation errors"""
        error_counts = {}
        for record in self.error_records:
            error_type = record['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_records),
            'error_types': error_counts,
            'most_common_error': max(error_counts.items(), key=lambda x: x[1]) if error_counts else None
        }

    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate data quality metrics"""
        try:
            total_validations = len(self.error_records) + 1  # Add 1 to avoid division by zero
            return {
                'success_rate': 1 - (len(self.error_records) / total_validations),
                'error_rate': len(self.error_records) / total_validations,
                'recovery_rate': sum(1 for r in self.error_records if r.get('recovered', False)) / total_validations
            }
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'success_rate': 0, 'error_rate': 1, 'recovery_rate': 0}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        quality_metrics = self._calculate_quality_metrics()
        
        if quality_metrics['error_rate'] > 0.1:
            recommendations.append("High error rate detected. Consider reviewing data sources.")
        
        if quality_metrics['recovery_rate'] < 0.8:
            recommendations.append("Low recovery rate. Consider implementing additional recovery methods.")
        
        error_summary = self._summarize_errors()
        if error_summary['most_common_error']:
            error_type, count = error_summary['most_common_error']
            recommendations.append(f"Frequent {error_type} errors. Consider implementing specific handling.")
        
        return recommendations