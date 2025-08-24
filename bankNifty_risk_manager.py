import numpy as np
from datetime import datetime, time
import pytz
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import json

ist = pytz.timezone('Asia/Kolkata')

class RiskManager:
    def __init__(self):
        self.daily_stats = self._initialize_daily_stats()
        self.risk_metrics = self._initialize_risk_metrics()
        self.position_limits = self._initialize_position_limits()
        self.ml_confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        self._load_risk_profile()

    def _initialize_daily_stats(self) -> Dict[str, Any]:
        """Initialize daily trading statistics"""
        return {
            'pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_portfolio_value': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'last_trade_time': None,
            'risk_used': 0.0
        }

    def _initialize_risk_metrics(self) -> Dict[str, float]:
        """Initialize risk monitoring metrics"""
        return {
            'current_risk_per_trade': Config.MAX_RISK_PER_TRADE,
            'available_risk': Config.MAX_RISK_PER_TRADE * Config.MAX_OPEN_TRADES,
            'volatility_factor': 1.0,
            'market_risk_factor': 1.0,
            'ml_confidence_factor': 1.0
        }

    def _initialize_position_limits(self) -> Dict[str, int]:
        """Initialize position size limits"""
        return {
            'max_position_size': Config.MAX_POSITION_QUANTITY,
            'min_position_size': Config.FIXED_QUANTITY,
            'current_positions': 0
        }

    def can_place_trade(self, ml_confidence: float, account_balance: float, 
                       current_positions: Dict, market_data: Dict) -> Dict[str, Any]:
        """
        Determine if a new trade can be placed based on various risk factors
        """
        try:
            current_time = datetime.now(ist).time()
            
            # Basic checks
            if not self._check_basic_conditions(current_time, account_balance):
                return {'can_trade': False, 'reason': 'basic_conditions_not_met'}

            # ML confidence check
            if not self._validate_ml_confidence(ml_confidence):
                return {'can_trade': False, 'reason': 'insufficient_ml_confidence'}

            # Risk budget check
            risk_check = self._check_risk_budget(account_balance)
            if not risk_check['can_trade']:
                return risk_check

            # Market condition check
            market_check = self._evaluate_market_conditions(market_data)
            if not market_check['can_trade']:
                return market_check

            # Position limits check
            if not self._check_position_limits(current_positions):
                return {'can_trade': False, 'reason': 'position_limit_reached'}

            # Calculate risk parameters
            risk_params = self._calculate_risk_parameters(
                ml_confidence, market_data, account_balance
            )

            return {
                'can_trade': True,
                'risk_params': risk_params,
                'confidence_level': self._get_confidence_level(ml_confidence)
            }

        except Exception as e:
            logger.error(f"Error in can_place_trade: {e}")
            logger.exception("Full traceback:")
            return {'can_trade': False, 'reason': 'error_in_risk_check'}

    def calculate_position_size(self, price: float, ml_confidence: float,
                              available_margin: float, market_data: Dict) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on ML confidence and risk parameters
        """
        try:
            # Base position calculation
            base_size = self._calculate_base_position_size(price, available_margin)
            
            # Adjust based on ML confidence
            confidence_multiplier = self._get_confidence_multiplier(ml_confidence)
            
            # Market condition adjustments
            market_multiplier = self._calculate_market_multiplier(market_data)
            
            # Volatility adjustments
            volatility_multiplier = self._calculate_volatility_multiplier(market_data)
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * market_multiplier * volatility_multiplier
            
            # Ensure within limits
            position_size = self._apply_position_limits(position_size)
            
            return {
                'position_size': position_size,
                'confidence_multiplier': confidence_multiplier,
                'market_multiplier': market_multiplier,
                'volatility_multiplier': volatility_multiplier
            }

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'position_size': Config.FIXED_QUANTITY}

    def calculate_risk_parameters(self, entry_price: float, ml_confidence: float,
                                market_data: Dict) -> Dict[str, Any]:
        """
        Calculate stop loss and target levels based on ML confidence
        """
        try:
            # Base risk parameters
            base_stop = Config.BASE_STOP_LOSS_PERCENT
            base_target = Config.BASE_TAKE_PROFIT_PERCENT
            
            # Adjust based on ML confidence
            confidence_level = self._get_confidence_level(ml_confidence)
            if confidence_level == 'high':
                stop_multiplier = 0.8  # Tighter stop for high confidence
                target_multiplier = 1.2  # Higher target for high confidence
            elif confidence_level == 'low':
                stop_multiplier = 1.2  # Wider stop for low confidence
                target_multiplier = 0.8  # Lower target for low confidence
            else:
                stop_multiplier = 1.0
                target_multiplier = 1.0
            
            # Adjust for market volatility
            volatility = market_data.get('volatility', 0.2)
            volatility_adjuster = volatility / 0.2  # Normalize around typical volatility
            
            final_stop = base_stop * stop_multiplier * volatility_adjuster
            final_target = base_target * target_multiplier * volatility_adjuster
            
            # Calculate actual prices
            stop_loss = entry_price * (1 - final_stop/100)
            target = entry_price * (1 + final_target/100)
            
            return {
                'stop_loss': stop_loss,
                'target': target,
                'stop_percent': final_stop,
                'target_percent': final_target,
                'trailing_stop': final_stop * 0.8,  # 80% of stop loss for trailing
                'confidence_level': confidence_level
            }

        except Exception as e:
            logger.error(f"Error calculating risk parameters: {e}")
            return {
                'stop_loss': entry_price * (1 - Config.BASE_STOP_LOSS_PERCENT/100),
                'target': entry_price * (1 + Config.BASE_TAKE_PROFIT_PERCENT/100),
                'stop_percent': Config.BASE_STOP_LOSS_PERCENT,
                'target_percent': Config.BASE_TAKE_PROFIT_PERCENT,
                'trailing_stop': Config.BASE_STOP_LOSS_PERCENT * 0.8,
                'confidence_level': 'medium'
            }

    def update_trade_metrics(self, trade_result: Dict[str, Any]) -> None:
        """
        Update risk metrics based on trade result
        """
        try:
            # Update daily stats
            self.daily_stats['total_trades'] += 1
            if trade_result['pnl'] > 0:
                self.daily_stats['winning_trades'] += 1
                self.daily_stats['largest_win'] = max(
                    self.daily_stats['largest_win'], 
                    trade_result['pnl']
                )
            else:
                self.daily_stats['losing_trades'] += 1
                self.daily_stats['largest_loss'] = min(
                    self.daily_stats['largest_loss'], 
                    trade_result['pnl']
                )
            
            # Update P&L
            self.daily_stats['pnl'] += trade_result['pnl']
            
            # Update drawdown
            if self.daily_stats['pnl'] > self.daily_stats['peak_portfolio_value']:
                self.daily_stats['peak_portfolio_value'] = self.daily_stats['pnl']
            current_drawdown = self.daily_stats['peak_portfolio_value'] - self.daily_stats['pnl']
            self.daily_stats['max_drawdown'] = max(
                self.daily_stats['max_drawdown'], 
                current_drawdown
            )
            
            # Update risk metrics
            self._adjust_risk_metrics(trade_result)
            
            # Save updated metrics
            self._save_risk_metrics()
            
        except Exception as e:
            logger.error(f"Error updating trade metrics: {e}")

    def _check_basic_conditions(self, current_time: time, account_balance: float) -> bool:
        """Check basic trading conditions"""
        try:
            # Time check
            if not self._is_valid_trading_time(current_time):
                logger.info("Outside valid trading hours")
                return False

            # Balance check
            if account_balance < Config.MIN_INVESTMENT:
                logger.info("Account balance below minimum requirement")
                return False

            # Daily loss limit check
            if self.daily_stats['pnl'] <= -Config.MAX_DAILY_LOSS:
                logger.info("Daily loss limit reached")
                return False

            # Daily profit target check
            if self.daily_stats['pnl'] >= Config.DAILY_PROFIT_LIMIT:
                logger.info("Daily profit target reached")
                return False

            return True

        except Exception as e:
            logger.error(f"Error in basic conditions check: {e}")
            return False

    def _validate_ml_confidence(self, confidence: float) -> bool:
        """Validate ML model confidence"""
        try:
            confidence_level = self._get_confidence_level(confidence)
            
            # Don't trade on low confidence unless other conditions are very favorable
            if confidence_level == 'low':
                return False
                
            # For medium confidence, check additional conditions
            if confidence_level == 'medium':
                return self._check_market_conditions_for_medium_confidence()
                
            return True

        except Exception as e:
            logger.error(f"Error validating ML confidence: {e}")
            return False

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category"""
        if confidence >= self.ml_confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.ml_confidence_thresholds['medium']:
            return 'medium'
        return 'low'

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get position size multiplier based on confidence"""
        try:
            if confidence >= self.ml_confidence_thresholds['high']:
                return 1.5
            elif confidence >= self.ml_confidence_thresholds['medium']:
                return 1.0
            return 0.5
        except Exception as e:
            logger.error(f"Error getting confidence multiplier: {e}")
            return 1.0

    def _calculate_market_multiplier(self, market_data: Dict) -> float:
        """Calculate market condition based position multiplier"""
        try:
            volatility = market_data.get('volatility', 0.2)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            # Reduce position size in high volatility
            if volatility > 0.3:
                return 0.8
            # Increase position size in strong trend
            elif trend_strength > 0.7:
                return 1.2
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating market multiplier: {e}")
            return 1.0

    def _calculate_volatility_multiplier(self, market_data: Dict) -> float:
        """Calculate volatility based position multiplier"""
        try:
            volatility = market_data.get('volatility', 0.2)
            base_volatility = 0.2  # Normal volatility level
            
            return base_volatility / volatility if volatility > 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0

    def _apply_position_limits(self, position_size: int) -> int:
        """Apply minimum and maximum position limits"""
        position_size = int(position_size)
        position_size = max(position_size, Config.FIXED_QUANTITY)
        position_size = min(position_size, Config.MAX_POSITION_QUANTITY)
        # Ensure multiple of lot size
        return (position_size // Config.LOT_SIZE) * Config.LOT_SIZE

    def _is_valid_trading_time(self, current_time: time) -> bool:
        """Check if current time is valid for trading"""
        return (Config.MARKET_OPEN_TIME <= current_time <= Config.MARKET_CLOSE_TIME and
                not Config.AVOID_TRADING_START <= current_time <= Config.AVOID_TRADING_END)

    def _check_risk_budget(self, account_balance: float) -> Dict[str, Any]:
        """Check if there's enough risk budget available"""
        try:
            used_risk = self.daily_stats['risk_used']
            max_risk = account_balance * Config.MAX_RISK_PERCENTAGE
            
            if used_risk >= max_risk:
                return {
                    'can_trade': False,
                    'reason': 'risk_budget_exceeded',
                    'used_risk': used_risk,
                    'max_risk': max_risk
                }
            
            return {
                'can_trade': True,
                'available_risk': max_risk - used_risk
            }
            
        except Exception as e:
            logger.error(f"Error checking risk budget: {e}")
            return {'can_trade': False, 'reason': 'risk_check_error'}

    def _evaluate_market_conditions(self, market_data: Dict) -> Dict[str, bool]:
        """Evaluate if market conditions are suitable for trading"""
        try:
            volatility = market_data.get('volatility', 0.2)
            
            # Don't trade in extremely high volatility
            if volatility > Config.HIGH_VOLATILITY_THRESHOLD:
                return {
                    'can_trade': False,
                    'reason': 'high_volatility'
                }
            
            return {'can_trade': True}
            
        except Exception as e:
            logger.error(f"Error evaluating market conditions: {e}")
            return {'can_trade': False, 'reason': 'market_evaluation_error'}

    def _adjust_risk_metrics(self, trade_result: Dict[str, Any]) -> None:
        """Adjust risk metrics based on trade result"""
        try:
            win_rate = (self.daily_stats['winning_trades'] / 
                       max(1, self.daily_stats['total_trades']))
            
            # Adjust risk per trade based on performance
            if win_rate > 0.6:
                self.risk_metrics['current_risk_per_trade'] = min(
                    self.risk_metrics['current_risk_per_trade'] * 1.1,
                    Config.MAX_RISK_PER_TRADE
                )
            elif win_rate < 0.4:
                self.risk_metrics['current_risk_per_trade'] *= 0.9
            
            # Adjust for drawdown
            if self.daily_stats['max_drawdown'] > Config.MAX_DAILY_LOSS * 0.5:
                self.risk_metrics['current_risk_per_trade'] *= 0.8
            
            # Update ML confidence factor based on prediction accuracy
            self._update_ml_confidence_factor(trade_result)
            
            logger.info(f"Adjusted risk metrics: {self.risk_metrics}")
            
        except Exception as e:
            logger.error(f"Error adjusting risk metrics: {e}")

    def _update_ml_confidence_factor(self, trade_result: Dict[str, Any]) -> None:
        """Update ML confidence factor based on prediction accuracy"""
        try:
            ml_confidence = trade_result.get('ml_confidence', 0.5)
            was_profitable = trade_result['pnl'] > 0
            
            # Update confidence thresholds based on performance
            if ml_confidence > self.ml_confidence_thresholds['high']:
                if not was_profitable:
                    self.ml_confidence_thresholds['high'] *= 1.05  # Increase threshold
            elif ml_confidence > self.ml_confidence_thresholds['medium']:
                if was_profitable:
                    self.ml_confidence_thresholds['medium'] *= 0.95  # Decrease threshold
            
            # Adjust confidence factor
            if was_profitable:
                self.risk_metrics['ml_confidence_factor'] = min(
                    self.risk_metrics['ml_confidence_factor'] * 1.1,
                    1.5
                )
            else:
                self.risk_metrics['ml_confidence_factor'] *= 0.9
            
        except Exception as e:
            logger.error(f"Error updating ML confidence factor: {e}")

    def calculate_dynamic_exits(self, position: Dict[str, Any], 
                              current_price: float, 
                              ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dynamic exit levels based on ML predictions"""
        try:
            entry_price = position['entry_price']
            original_stop = position['stop_loss']
            original_target = position['target']
            
            # Get new ML confidence
            new_confidence = ml_prediction.get('confidence', 0.5)
            price_prediction = ml_prediction.get('predicted_price')
            
            # Adjust stops based on ML confidence change
            if new_confidence < position.get('entry_confidence', 0.5):
                # Tighten stops if confidence decreases
                new_stop = self._calculate_tighter_stop(
                    current_price, original_stop, entry_price
                )
            else:
                # Consider widening stops if confidence increases
                new_stop = self._calculate_wider_stop(
                    current_price, original_stop, entry_price
                )
            
            # Adjust targets based on ML price prediction
            if price_prediction:
                new_target = self._calculate_dynamic_target(
                    current_price, original_target, price_prediction
                )
            else:
                new_target = original_target
            
            return {
                'stop_loss': new_stop,
                'target': new_target,
                'trailing_stop': self._calculate_trailing_stop(
                    current_price, new_stop, position
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic exits: {e}")
            return {
                'stop_loss': position['stop_loss'],
                'target': position['target'],
                'trailing_stop': position.get('trailing_stop')
            }

    def _calculate_tighter_stop(self, current_price: float, 
                              original_stop: float, 
                              entry_price: float) -> float:
        """Calculate tighter stop loss level"""
        try:
            current_stop_distance = abs(current_price - original_stop)
            new_stop_distance = current_stop_distance * 0.8  # Tighten by 20%
            
            if current_price > entry_price:  # Long position
                return current_price - new_stop_distance
            else:  # Short position
                return current_price + new_stop_distance
                
        except Exception as e:
            logger.error(f"Error calculating tighter stop: {e}")
            return original_stop

    def _calculate_wider_stop(self, current_price: float, 
                            original_stop: float, 
                            entry_price: float) -> float:
        """Calculate wider stop loss level"""
        try:
            current_stop_distance = abs(current_price - original_stop)
            max_stop_distance = abs(entry_price - original_stop) * 1.5
            new_stop_distance = min(
                current_stop_distance * 1.2,  # Widen by 20%
                max_stop_distance  # But don't exceed maximum
            )
            
            if current_price > entry_price:
                return current_price - new_stop_distance
            else:
                return current_price + new_stop_distance
                
        except Exception as e:
            logger.error(f"Error calculating wider stop: {e}")
            return original_stop

    def _calculate_dynamic_target(self, current_price: float, 
                                original_target: float, 
                                predicted_price: float) -> float:
        """Calculate dynamic target based on ML price prediction"""
        try:
            # Use predicted price if it's more favorable than original target
            if current_price > original_target:  # Long position
                return max(original_target, predicted_price)
            else:  # Short position
                return min(original_target, predicted_price)
                
        except Exception as e:
            logger.error(f"Error calculating dynamic target: {e}")
            return original_target

    def _calculate_trailing_stop(self, current_price: float, 
                               stop_loss: float, 
                               position: Dict[str, Any]) -> float:
        """Calculate trailing stop level"""
        try:
            if not position.get('trailing_active'):
                return stop_loss
            
            profit_percent = (current_price - position['entry_price']) / \
                           position['entry_price'] * 100
                           
            if profit_percent > Config.TRAILING_STOP_ACTIVATION_PERCENT:
                trail_percent = Config.TRAILING_STOP_DISTANCE_PERCENT
                if current_price > position['entry_price']:
                    return max(
                        stop_loss,
                        current_price * (1 - trail_percent/100)
                    )
                else:
                    return min(
                        stop_loss,
                        current_price * (1 + trail_percent/100)
                    )
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return stop_loss

    def _load_risk_profile(self) -> None:
        """Load saved risk profile from file"""
        try:
            risk_file = Path(Config.DATA_DIR) / 'risk_profile.json'
            if risk_file.exists():
                with open(risk_file, 'r') as f:
                    risk_data = json.load(f)
                    self.ml_confidence_thresholds = risk_data.get(
                        'ml_confidence_thresholds',
                        self.ml_confidence_thresholds
                    )
                    self.risk_metrics = risk_data.get(
                        'risk_metrics',
                        self.risk_metrics
                    )
                    logger.info("Loaded risk profile from file")
                    
        except Exception as e:
            logger.error(f"Error loading risk profile: {e}")

    def _save_risk_metrics(self) -> None:
        """Save current risk metrics to file"""
        try:
            risk_data = {
                'ml_confidence_thresholds': self.ml_confidence_thresholds,
                'risk_metrics': self.risk_metrics,
                'daily_stats': self.daily_stats,
                'last_updated': datetime.now(ist).isoformat()
            }
            
            risk_file = Path(Config.DATA_DIR) / 'risk_profile.json'
            risk_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(risk_file, 'w') as f:
                json.dump(risk_data, f, indent=2)
                
            logger.info("Saved risk metrics to file")
            
        except Exception as e:
            logger.error(f"Error saving risk metrics: {e}")

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            win_rate = (self.daily_stats['winning_trades'] / 
                       max(1, self.daily_stats['total_trades']))
            
            return {
                'daily_stats': self.daily_stats,
                'risk_metrics': self.risk_metrics,
                'position_limits': self.position_limits,
                'ml_confidence_thresholds': self.ml_confidence_thresholds,
                'performance_metrics': {
                    'win_rate': win_rate,
                    'risk_reward_ratio': abs(
                        self.daily_stats['largest_win'] / 
                        self.daily_stats['largest_loss']
                    ) if self.daily_stats['largest_loss'] != 0 else float('inf'),
                    'max_drawdown_percentage': (
                        self.daily_stats['max_drawdown'] / 
                        self.daily_stats['peak_portfolio_value'] * 100
                    ) if self.daily_stats['peak_portfolio_value'] != 0 else 0
                },
                'timestamp': datetime.now(ist).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}

    def reset_daily_metrics(self) -> None:
        """Reset daily trading metrics"""
        try:
            self.daily_stats = self._initialize_daily_stats()
            self.risk_metrics['current_risk_per_trade'] = Config.MAX_RISK_PER_TRADE
            logger.info("Reset daily risk metrics")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {e}")

if __name__ == "__main__":
    try:
        # Test code
        risk_manager = RiskManager()
        # Add test scenarios here
        pass
    except Exception as e:
        logger.error(f"Error in risk manager test: {e}")