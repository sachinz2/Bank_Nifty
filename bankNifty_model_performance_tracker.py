import pandas as pd
import numpy as np
from datetime import datetime
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import json
from pathlib import Path

class ModelPerformanceTracker:
    def __init__(self):
        self.predictions = []
        self.actual_outcomes = []
        self.trade_results = []
        self.performance_history = []
        self.current_trade_stats = {
            'wins': 0,
            'losses': 0,
            'total_profit': 0,
            'total_loss': 0
        }
    
    def add_prediction(self, prediction, actual_outcome, trade_result):
        """Add a new prediction and its outcome"""
        self.predictions.append(prediction)
        self.actual_outcomes.append(actual_outcome)
        self.trade_results.append(trade_result)
        
        # Update trade statistics
        if trade_result > 0:
            self.current_trade_stats['wins'] += 1
            self.current_trade_stats['total_profit'] += trade_result
        elif trade_result < 0:
            self.current_trade_stats['losses'] += 1
            self.current_trade_stats['total_loss'] += abs(trade_result)
            
        # Save performance metrics periodically
        if len(self.predictions) % Config.PERFORMANCE_EVALUATION_INTERVAL == 0:
            self.save_performance_metrics()
    
    def calculate_metrics(self, window=100):
        """Calculate performance metrics for recent trades"""
        if len(self.predictions) < window:
            return None
            
        recent_predictions = self.predictions[-window:]
        recent_outcomes = self.actual_outcomes[-window:]
        recent_results = self.trade_results[-window:]
        
        metrics = {
            'accuracy': np.mean(np.array(recent_predictions) == np.array(recent_outcomes)),
            'win_rate': self.current_trade_stats['wins'] / (self.current_trade_stats['wins'] + self.current_trade_stats['losses']) if (self.current_trade_stats['wins'] + self.current_trade_stats['losses']) > 0 else 0,
            'profit_factor': self.current_trade_stats['total_profit'] / abs(self.current_trade_stats['total_loss']) if self.current_trade_stats['total_loss'] != 0 else float('inf'),
            'average_win': self.current_trade_stats['total_profit'] / self.current_trade_stats['wins'] if self.current_trade_stats['wins'] > 0 else 0,
            'average_loss': self.current_trade_stats['total_loss'] / self.current_trade_stats['losses'] if self.current_trade_stats['losses'] > 0 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(recent_results),
            'max_drawdown': self.calculate_max_drawdown(recent_results)
        }
        
        return metrics
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
            
        returns_series = pd.Series(returns)
        excess_returns = returns_series - (risk_free_rate / 252)  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if not returns:
            return 0
            
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(min(drawdown, default=0))
    
    def save_performance_metrics(self):
        """Save performance metrics to file"""
        try:
            metrics = self.calculate_metrics()
            if metrics:
                metrics['timestamp'] = datetime.now().isoformat()
                self.performance_history.append(metrics)
                
                # Save to file
                performance_file = Path(Config.DATA_DIR) / 'model_performance.json'
                performance_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(performance_file, 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
                
                logger.info(f"Saved performance metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            logger.exception("Full traceback:")
    
    def should_retrain(self, threshold=0.55):
        """Determine if model retraining is needed"""
        metrics = self.calculate_metrics()
        if metrics and metrics['accuracy'] < threshold:
            logger.info(f"Model accuracy ({metrics['accuracy']:.2f}) below threshold ({threshold}). Retraining recommended.")
            return True
        return False
    
    def get_performance_summary(self):
        """Get a summary of current performance"""
        metrics = self.calculate_metrics()
        if not metrics:
            return "Insufficient data for performance summary"
            
        summary = (
            f"Performance Summary:\n"
            f"Accuracy: {metrics['accuracy']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Average Win: ₹{metrics['average_win']:.2f}\n"
            f"Average Loss: ₹{metrics['average_loss']:.2f}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: ₹{metrics['max_drawdown']:.2f}"
        )
        
        return summary