import pandas as pd
import numpy as np
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import pytz
from datetime import datetime

ist = pytz.timezone('Asia/Kolkata')

class StrategyEvaluator:
    def __init__(self):
        self.strategy_performance = {}
        self.current_trades = {}
        self.historical_trades = []

    def evaluate_entry_conditions(self, market_analysis, xgb_prediction, technical_signals):
        """
        Evaluate if entry conditions are met
        """
        try:
            # Calculate composite score
            score = self._calculate_entry_score(market_analysis, xgb_prediction, technical_signals)

            # Evaluate market conditions
            market_suitable = self._evaluate_market_conditions(market_analysis)

            # Check signal strength
            signals_valid = self._validate_signals(xgb_prediction, technical_signals)

            # Final decision
            entry_decision = {
                'should_enter': score >= Config.ENTRY_THRESHOLD and market_suitable and signals_valid,
                'score': score,
                'market_suitable': market_suitable,
                'signals_valid': signals_valid,
                'timestamp': datetime.now(ist)
            }

            logger.info(f"Entry evaluation: {entry_decision}")
            return entry_decision

        except Exception as e:
            logger.error(f"Error evaluating entry conditions: {e}")
            return {'should_enter': False, 'score': 0, 'market_suitable': False, 'signals_valid': False}

    def _calculate_entry_score(self, market_analysis, xgb_prediction, technical_signals):
        """
        Calculate composite entry score
        """
        try:
            score = 0

            # Market analysis component (0-40 points)
            if market_analysis['trend'] in ['strong_uptrend', 'strong_downtrend']:
                score += 40
            elif market_analysis['trend'] in ['uptrend', 'downtrend']:
                score += 30
            
            # XGBoost prediction component (0-30 points)
            prediction_confidence = max(xgb_prediction['buy_probability'], xgb_prediction['sell_probability'])
            score += prediction_confidence * 30

            # Technical signals component (0-30 points)
            technical_score = sum(signal['strength'] for signal in technical_signals.values())
            score += min(technical_score, 30)

            return score

        except Exception as e:
            logger.error(f"Error calculating entry score: {e}")
            return 0

    def _evaluate_market_conditions(self, market_analysis):
        """
        Evaluate if market conditions are suitable for trading
        """
        try:
            # Check volatility
            if market_analysis['volatility'] == 'high' and market_analysis['strength'] < 60:
                return False

            # Check trend stability
            if market_analysis['trend'] == 'consolidation':
                return False

            # Check volume
            if market_analysis['volume_trend'] == 'low':
                return False

            return True

        except Exception as e:
            logger.error(f"Error evaluating market conditions: {e}")
            return False

    def _validate_signals(self, xgb_prediction, technical_signals):
        """
        Validate signal strength and consistency
        """
        try:
            # Check XGBoost prediction confidence
            prediction_confidence = max(xgb_prediction['buy_probability'], xgb_prediction['sell_probability'])
            if prediction_confidence < Config.ML_BUY_THRESHOLD:
                return False

            # Check technical signals consistency
            signal_count = sum(1 for signal in technical_signals.values() if signal['valid'])
            min_signals_required = len(technical_signals) * 0.6
            
            return signal_count >= min_signals_required

        except Exception as e:
            logger.error(f"Error validating signals: {e}")
            return False

    def evaluate_exit_conditions(self, position, current_price, market_analysis, xgb_prediction):
        """
        Evaluate if exit conditions are met
        """
        try:
            # Calculate profit/loss
            pnl_percent = self._calculate_pnl_percent(position, current_price)

            # Check stop loss
            if pnl_percent <= -Config.STOP_LOSS_PERCENT:
                return {'should_exit': True, 'reason': 'stop_loss'}

            # Check take profit
            if pnl_percent >= Config.TAKE_PROFIT_PERCENT:
                return {'should_exit': True, 'reason': 'take_profit'}

            # Check trailing stop
            if self._check_trailing_stop(position, current_price):
                return {'should_exit': True, 'reason': 'trailing_stop'}

            # Check trend reversal
            if self._check_trend_reversal(position, market_analysis):
                return {'should_exit': True, 'reason': 'trend_reversal'}

            # Check model prediction reversal
            if self._check_prediction_reversal(position, xgb_prediction):
                return {'should_exit': True, 'reason': 'prediction_reversal'}

            return {'should_exit': False, 'reason': None}

        except Exception as e:
            logger.error(f"Error evaluating exit conditions: {e}")
            return {'should_exit': False, 'reason': None}

    def _calculate_pnl_percent(self, position, current_price):
        """
        Calculate profit/loss percentage
        """
        try:
            entry_price = position['average_price']
            if position['quantity'] > 0:
                return ((current_price - entry_price) / entry_price) * 100
            else:
                return ((entry_price - current_price) / entry_price) * 100
        except Exception as e:
            logger.error(f"Error calculating P&L percent: {e}")
            return 0

    def _check_trailing_stop(self, position, current_price):
        """
        Check if trailing stop loss is hit
        """
        try:
            if position['peak_profit_percent'] >= Config.TRAILING_STOP_ACTIVATION_PERCENT:
                current_profit = self._calculate_pnl_percent(position, current_price)
                trailing_stop_level = position['peak_profit_percent'] - Config.TRAILING_STOP_DISTANCE_PERCENT
                return current_profit < trailing_stop_level
            return False
        except Exception as e:
            logger.error(f"Error checking trailing stop: {e}")
            return False

    def _check_trend_reversal(self, position, market_analysis):
        """
        Check for trend reversal
        """
        try:
            if position['quantity'] > 0:
                return market_analysis['trend'] in ['strong_downtrend', 'downtrend']
            else:
                return market_analysis['trend'] in ['strong_uptrend', 'uptrend']
        except Exception as e:
            logger.error(f"Error checking trend reversal: {e}")
            return False

    def _check_prediction_reversal(self, position, xgb_prediction):
        """
        Check for model prediction reversal
        """
        try:
            if position['quantity'] > 0:
                return xgb_prediction['sell_probability'] > Config.ML_SELL_THRESHOLD
            else:
                return xgb_prediction['buy_probability'] > Config.ML_BUY_THRESHOLD
        except Exception as e:
            logger.error(f"Error checking prediction reversal: {e}")
            return False

    def update_strategy_performance(self, trade_result):
        """
        Update strategy performance metrics
        """
        try:
            self.historical_trades.append(trade_result)
            
            # Update performance metrics
            self.strategy_performance.update({
                'total_trades': len(self.historical_trades),
                'winning_trades': sum(1 for trade in self.historical_trades if trade['pnl'] > 0),
                'losing_trades': sum(1 for trade in self.historical_trades if trade['pnl'] < 0),
                'total_pnl': sum(trade['pnl'] for trade in self.historical_trades),
                'max_profit': max((trade['pnl'] for trade in self.historical_trades), default=0),
                'max_loss': min((trade['pnl'] for trade in self.historical_trades), default=0),
                'average_profit': np.mean([trade['pnl'] for trade in self.historical_trades if trade['pnl'] > 0]) if any(trade['pnl'] > 0 for trade in self.historical_trades) else 0,
                'average_loss': np.mean([trade['pnl'] for trade in self.historical_trades if trade['pnl'] < 0]) if any(trade['pnl'] < 0 for trade in self.historical_trades) else 0,
                'profit_factor': self._calculate_profit_factor(),
                'win_rate': self._calculate_win_rate(),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(),
                'max_drawdown': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'last_updated': datetime.now(ist)
            })

            logger.info(f"Updated strategy performance: {self.strategy_performance}")

        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
            logger.exception("Full traceback:")

    def _calculate_profit_factor(self):
        """Calculate profit factor"""
        try:
            total_profit = sum(trade['pnl'] for trade in self.historical_trades if trade['pnl'] > 0)
            total_loss = abs(sum(trade['pnl'] for trade in self.historical_trades if trade['pnl'] < 0))
            return total_profit / total_loss if total_loss != 0 else float('inf')
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0

    def _calculate_win_rate(self):
        """Calculate win rate"""
        try:
            if not self.historical_trades:
                return 0
            winning_trades = sum(1 for trade in self.historical_trades if trade['pnl'] > 0)
            return (winning_trades / len(self.historical_trades)) * 100
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0

    def _calculate_risk_reward_ratio(self):
        """Calculate risk-reward ratio"""
        try:
            avg_profit = np.mean([trade['pnl'] for trade in self.historical_trades if trade['pnl'] > 0]) if any(trade['pnl'] > 0 for trade in self.historical_trades) else 0
            avg_loss = abs(np.mean([trade['pnl'] for trade in self.historical_trades if trade['pnl'] < 0])) if any(trade['pnl'] < 0 for trade in self.historical_trades) else 1
            return avg_profit / avg_loss if avg_loss != 0 else float('inf')
        except Exception as e:
            logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        try:
            if not self.historical_trades:
                return 0
            
            cumulative_pnl = np.cumsum([trade['pnl'] for trade in self.historical_trades])
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            return float(max(drawdown, default=0))
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0

    def _calculate_sharpe_ratio(self, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        try:
            if not self.historical_trades:
                return 0
            
            returns = [trade['pnl'] for trade in self.historical_trades]
            excess_returns = np.array(returns) - (risk_free_rate / 252)  # Daily risk-free rate
            if len(excess_returns) < 2:
                return 0
                
            return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0

    def evaluate_strategy_health(self):
        """
        Evaluate overall strategy health
        """
        try:
            if not self.strategy_performance:
                return {'status': 'insufficient_data', 'score': 0}

            health_score = 0
            
            # Win rate component (0-30 points)
            win_rate = self.strategy_performance.get('win_rate', 0)
            health_score += min(win_rate / 2, 30)  # Max 30 points for 60% win rate

            # Profit factor component (0-30 points)
            profit_factor = self.strategy_performance.get('profit_factor', 0)
            health_score += min(profit_factor * 15, 30)  # Max 30 points for profit factor of 2

            # Risk-reward component (0-20 points)
            risk_reward = self.strategy_performance.get('risk_reward_ratio', 0)
            health_score += min(risk_reward * 10, 20)  # Max 20 points for RR ratio of 2

            # Sharpe ratio component (0-20 points)
            sharpe_ratio = self.strategy_performance.get('sharpe_ratio', 0)
            health_score += min(sharpe_ratio * 10, 20)  # Max 20 points for Sharpe ratio of 2

            # Determine status
            status = 'healthy' if health_score >= 70 else 'moderate' if health_score >= 50 else 'unhealthy'

            return {
                'status': status,
                'score': health_score,
                'metrics': {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'risk_reward': risk_reward,
                    'sharpe_ratio': sharpe_ratio
                },
                'timestamp': datetime.now(ist)
            }

        except Exception as e:
            logger.error(f"Error evaluating strategy health: {e}")
            return {'status': 'error', 'score': 0}

    def should_continue_trading(self):
        """
        Determine if trading should continue based on strategy health
        """
        try:
            health = self.evaluate_strategy_health()
            
            if health['status'] == 'unhealthy':
                return False
                
            if health['score'] < 50:
                logger.warning(f"Strategy health score ({health['score']}) below threshold")
                return False

            if self.strategy_performance.get('max_drawdown', 0) > Config.MAX_DRAWDOWN_LIMIT:
                logger.warning("Maximum drawdown limit exceeded")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking if trading should continue: {e}")
            return False

    def get_performance_report(self):
        """
        Generate comprehensive performance report
        """
        try:
            report = {
                'summary': self.strategy_performance,
                'health': self.evaluate_strategy_health(),
                'recent_trades': self.historical_trades[-10:] if self.historical_trades else [],
                'timestamp': datetime.now(ist)
            }
            
            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return None