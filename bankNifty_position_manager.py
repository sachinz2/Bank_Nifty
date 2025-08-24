import asyncio
from datetime import datetime
import pytz
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json

ist = pytz.timezone('Asia/Kolkata')

class PositionManager:
    def __init__(self):
        self.positions = {}
        self.position_history = []
        self.position_lock = asyncio.Lock()
        self.total_pnl = 0
        self.peak_pnl = 0
        self.max_drawdown = 0
        self._load_position_history()

    async def update_position(self,
                            symbol: str,
                            quantity: int,
                            entry_price: float,
                            ml_confidence: float,
                            risk_params: Dict[str, Any]) -> bool:
        """Update position with ML-based parameters"""
        try:
            async with self.position_lock:
                if symbol not in self.positions:
                    # New position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'current_price': entry_price,
                        'ml_confidence': ml_confidence,
                        'entry_time': datetime.now(ist).isoformat(),
                        'peak_price': entry_price,
                        'lowest_price': entry_price,
                        'stop_loss': risk_params['stop_loss'],
                        'target': risk_params['target'],
                        'trailing_stop': risk_params['trailing_stop'],
                        'pnl': 0,
                        'peak_pnl': 0,
                        'status': 'ACTIVE',
                        'modifications': []
                    }
                else:
                    # Existing position
                    position = self.positions[symbol]
                    avg_entry = ((position['entry_price'] * position['quantity']) +
                               (entry_price * quantity)) / (position['quantity'] + quantity)
                    
                    position.update({
                        'quantity': position['quantity'] + quantity,
                        'entry_price': avg_entry,
                        'modifications': position['modifications'] + [{
                            'time': datetime.now(ist).isoformat(),
                            'type': 'ADD',
                            'quantity': quantity,
                            'price': entry_price
                        }]
                    })

                await self._save_position_state()
                return True

        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            logger.exception("Full traceback:")
            return False

    async def update_position_parameters(self,
                                       symbol: str,
                                       current_price: float,
                                       ml_prediction: Dict[str, Any]) -> None:
        """Update position parameters based on ML prediction"""
        try:
            async with self.position_lock:
                if symbol not in self.positions:
                    return

                position = self.positions[symbol]
                
                # Update price information
                position['current_price'] = current_price
                position['peak_price'] = max(position['peak_price'], current_price)
                position['lowest_price'] = min(position['lowest_price'], current_price)

                # Calculate current P&L
                position['pnl'] = self._calculate_pnl(position, current_price)
                position['peak_pnl'] = max(position['peak_pnl'], position['pnl'])

                # Update stops based on ML confidence
                self._update_position_stops(position, current_price, ml_prediction)

                await self._save_position_state()

        except Exception as e:
            logger.error(f"Error updating position parameters for {symbol}: {e}")

    async def close_position(self,
                           symbol: str,
                           exit_price: float,
                           exit_reason: str) -> bool:
        """Close position and record results"""
        try:
            async with self.position_lock:
                if symbol not in self.positions:
                    return False

                position = self.positions[symbol]
                final_pnl = self._calculate_pnl(position, exit_price)
                
                # Record position result
                position_result = {
                    **position,
                    'exit_price': exit_price,
                    'exit_time': datetime.now(ist).isoformat(),
                    'exit_reason': exit_reason,
                    'final_pnl': final_pnl,
                    'status': 'CLOSED',
                    'holding_duration': self._calculate_holding_duration(position['entry_time'])
                }

                # Update tracking metrics
                self.total_pnl += final_pnl
                self.peak_pnl = max(self.peak_pnl, self.total_pnl)
                current_drawdown = self.peak_pnl - self.total_pnl
                self.max_drawdown = max(self.max_drawdown, current_drawdown)

                # Add to history and remove from active positions
                self.position_history.append(position_result)
                del self.positions[symbol]

                await self._save_position_state()
                return True

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False

    def get_position_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position status"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions"""
        return self.positions

    def get_position_history(self) -> List[Dict[str, Any]]:
        """Get complete position history"""
        return self.position_history

    def get_total_exposure(self) -> float:
        """Calculate total position exposure"""
        try:
            return sum(pos['quantity'] * pos['current_price'] 
                      for pos in self.positions.values())
        except Exception as e:
            logger.error(f"Error calculating total exposure: {e}")
            return 0.0

    def get_total_pnl(self) -> float:
        """Get total P&L across all positions"""
        return self.total_pnl

    def _calculate_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """Calculate position P&L"""
        try:
            return position['quantity'] * (current_price - position['entry_price'])
        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0

    def _update_position_stops(self,
                             position: Dict[str, Any],
                             current_price: float,
                             ml_prediction: Dict[str, Any]) -> None:
        """Update position stop levels based on ML prediction"""
        try:
            # Get new confidence level
            new_confidence = ml_prediction.get('confidence', 0.5)
            original_confidence = position['ml_confidence']

            # Adjust stops based on confidence change
            if new_confidence < original_confidence:
                # Tighten stops if confidence decreases
                position['stop_loss'] = self._calculate_tighter_stop(
                    current_price, position['stop_loss'], position['entry_price']
                )
            elif new_confidence > original_confidence:
                # Consider widening stops if confidence increases
                position['stop_loss'] = self._calculate_wider_stop(
                    current_price, position['stop_loss'], position['entry_price']
                )

            # Update trailing stop
            if position['pnl'] > 0:
                position['trailing_stop'] = self._calculate_trailing_stop(
                    current_price, position['stop_loss'], new_confidence
                )

            position['ml_confidence'] = new_confidence
            
        except Exception as e:
            logger.error(f"Error updating position stops: {e}")

    def _calculate_tighter_stop(self,
                              current_price: float,
                              current_stop: float,
                              entry_price: float) -> float:
        """Calculate tighter stop loss"""
        try:
            current_stop_distance = abs(current_price - current_stop)
            new_stop_distance = current_stop_distance * 0.8  # Tighten by 20%
            
            if current_price > entry_price:  # Long position
                return max(current_price - new_stop_distance, entry_price)
            else:  # Short position
                return min(current_price + new_stop_distance, entry_price)
                
        except Exception as e:
            logger.error(f"Error calculating tighter stop: {e}")
            return current_stop

    def _calculate_wider_stop(self,
                            current_price: float,
                            current_stop: float,
                            entry_price: float) -> float:
        """Calculate wider stop loss"""
        try:
            current_stop_distance = abs(current_price - current_stop)
            max_stop_distance = abs(entry_price - current_stop) * 1.5
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
            return current_stop

    def _calculate_trailing_stop(self,
                               current_price: float,
                               base_stop: float,
                               ml_confidence: float) -> float:
        """Calculate trailing stop based on ML confidence"""
        try:
            trail_percentage = Config.BASE_TRAILING_STOP_PERCENT
            
            # Adjust trail percentage based on ML confidence
            if ml_confidence >= 0.8:
                trail_percentage *= 0.8  # Tighter trail for high confidence
            elif ml_confidence < 0.5:
                trail_percentage *= 1.2  # Wider trail for low confidence
                
            trail_amount = current_price * (trail_percentage / 100)
            return current_price - trail_amount
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return base_stop

    def _calculate_holding_duration(self, entry_time: str) -> str:
        """Calculate position holding duration"""
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            duration = datetime.now(ist) - entry_dt
            return str(duration)
        except Exception as e:
            logger.error(f"Error calculating holding duration: {e}")
            return "Unknown"

    async def _save_position_state(self) -> None:
        """Save position state to file"""
        try:
            state = {
                'active_positions': self.positions,
                'position_history': self.position_history,
                'total_pnl': self.total_pnl,
                'peak_pnl': self.peak_pnl,
                'max_drawdown': self.max_drawdown,
                'last_updated': datetime.now(ist).isoformat()
            }
            
            state_file = Path(Config.DATA_DIR) / 'position_state.json'
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving position state: {e}")

    def _load_position_history(self) -> None:
        """Load position history from file"""
        try:
            state_file = Path(Config.DATA_DIR) / 'position_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.position_history = state.get('position_history', [])
                    self.total_pnl = state.get('total_pnl', 0)
                    self.peak_pnl = state.get('peak_pnl', 0)
                    self.max_drawdown = state.get('max_drawdown', 0)
                    
        except Exception as e:
            logger.error(f"Error loading position history: {e}")

    def get_position_metrics(self) -> Dict[str, Any]:
        """Get comprehensive position metrics"""
        try:
            wins = len([p for p in self.position_history if p['final_pnl'] > 0])
            losses = len([p for p in self.position_history if p['final_pnl'] <= 0])
            total_trades = len(self.position_history)
            
            return {
                'total_trades': total_trades,
                'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
                'total_pnl': self.total_pnl,
                'peak_pnl': self.peak_pnl,
                'max_drawdown': self.max_drawdown,
                'active_positions': len(self.positions),
                'total_exposure': self.get_total_exposure(),
                'average_position_size': self._calculate_avg_position_size(),
                'average_holding_time': self._calculate_avg_holding_time()
            }
            
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return {}

    def _calculate_avg_position_size(self) -> float:
        """Calculate average position size"""
        try:
            if not self.position_history:
                return 0.0
            return sum(p['quantity'] for p in self.position_history) / len(self.position_history)
        except Exception as e:
            logger.error(f"Error calculating average position size: {e}")
            return 0.0

    def _calculate_avg_holding_time(self) -> float:
        """Calculate average position holding time"""
        try:
            if not self.position_history:
                return 0.0
                
            holding_times = []
            for position in self.position_history:
                entry_time = datetime.fromisoformat(position['entry_time'])
                exit_time = datetime.fromisoformat(position['exit_time'])
                holding_time = (exit_time - entry_time).total_seconds() / 3600  # in hours
                holding_times.append(holding_time)
                
            return sum(holding_times) / len(holding_times)
            
        except Exception as e:
            logger.error(f"Error calculating average holding time: {e}")
            return 0.0

if __name__ == "__main__":
    # Test code
    pass