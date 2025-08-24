import asyncio
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from logging_utils import banknifty_logger as logger
from kiteconnect_common import get_kite
from bankNifty_config import Config
from bankNifty_ML_model import MLDecisionMaker
from bankNifty_data_fetcher import DataFetcher
from bankNifty_feature_engineering import FeatureEngineer
from bankNifty_risk_manager import RiskManager
from bankNifty_position_manager import PositionManager
from bankNifty_order_executor import OrderExecutor
from model_performance_tracker import ModelPerformanceTracker
from bankNifty_notifications import NotificationManager
from pathlib import Path
import traceback
from typing import Dict, Any, Optional

ist = pytz.timezone('Asia/Kolkata')

class BankNiftyTrader:
    def __init__(self):
        self.initialize_components()
        self.running = False
        self.last_data_update = None
        self.last_model_retrain = None
        self.performance_metrics = {}

    def initialize_components(self) -> None:
        """Initialize all trading system components"""
        try:
            logger.info("Initializing trading system components...")
            
            # Initialize Kite connection
            self.kite = get_kite()
            if not self.kite:
                raise Exception("Failed to initialize Kite connection")

            # Initialize core components
            self.ml_model = MLDecisionMaker()
            self.data_fetcher = DataFetcher(Config.KITE_API_KEY, self.kite.access_token)
            self.feature_engineer = FeatureEngineer()
            self.risk_manager = RiskManager()
            self.position_manager = PositionManager()
            self.order_executor = OrderExecutor()
            self.performance_tracker = ModelPerformanceTracker()
            self.notification_manager = NotificationManager()

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.exception("Full traceback:")
            raise

    async def start_trading(self) -> None:
        """Start the trading system"""
        try:
            logger.info("Starting BankNifty trading system...")
            self.running = True
            
            while self.running:
                if not await self._should_continue_trading():
                    await self._handle_trading_pause()
                    continue

                try:
                    # Fetch and process market data
                    market_data = await self._update_market_data()
                    if not market_data:
                        continue

                    # Generate ML features and get decision
                    features = self._prepare_features(market_data)
                    trading_decision = await self._get_trading_decision(features)

                    # Execute trading logic
                    await self._execute_trading_logic(trading_decision, market_data)

                    # Monitor and manage positions
                    await self._manage_positions(market_data)

                    # Update performance metrics
                    self._update_performance_metrics()

                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    logger.exception("Full traceback:")
                    await self._handle_error()

                await asyncio.sleep(Config.TRADING_INTERVAL)

        except Exception as e:
            logger.error(f"Fatal error in trading system: {e}")
            logger.exception("Full traceback:")
            await self.stop_trading()

    async def _should_continue_trading(self) -> bool:
        """Determine if trading should continue"""
        try:
            current_time = datetime.now(ist).time()
            
            # Check trading hours
            if not (Config.MARKET_OPEN_TIME <= current_time <= Config.MARKET_CLOSE_TIME):
                logger.info("Outside trading hours")
                return False

            # Check system health
            if not self._check_system_health():
                logger.warning("System health check failed")
                return False

            # Check performance metrics
            if not self.performance_tracker.should_continue_trading():
                logger.warning("Performance metrics below threshold")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking trading conditions: {e}")
            return False

    async def _update_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetch and update market data"""
        try:
            current_time = datetime.now(ist)
            
            # Check if update is needed
            if (self.last_data_update and 
                (current_time - self.last_data_update).seconds < Config.DATA_UPDATE_INTERVAL):
                return self.data_fetcher.get_cached_data()

            # Fetch new data
            market_data = await self.data_fetcher.fetch_all_data()
            if market_data:
                self.last_data_update = current_time
                return market_data

            logger.error("Failed to fetch market data")
            return None

        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return None

    def _prepare_features(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            features = self.feature_engineer.engineer_features(
                historical_data=market_data['historical_data'],
                options_data=market_data['options_chain'],
                sentiment_data=market_data['sentiment_data'],
                market_breadth_data=market_data['market_breadth'],
                volatility_data=market_data['volatility_data']
            )
            
            return self.feature_engineer.preprocess_features(features)

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    async def _get_trading_decision(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get trading decision from ML model"""
        try:
            if features.empty:
                return {'action': 'hold', 'confidence': 0}

            # Get ML model decision
            decision = self.ml_model.make_decision(features)
            
            # Log decision details
            logger.info(f"ML Model Decision: {decision}")
            
            return decision

        except Exception as e:
            logger.error(f"Error getting trading decision: {e}")
            return {'action': 'hold', 'confidence': 0}

    async def _execute_trading_logic(self, decision: Dict[str, Any], 
                                   market_data: Dict[str, Any]) -> None:
        """Execute trading logic based on ML decision"""
        try:
            # Check if we can trade
            account_balance = await self._get_account_balance()
            current_positions = self.position_manager.get_positions()
            
            can_trade = self.risk_manager.can_place_trade(
                ml_confidence=decision['confidence'],
                account_balance=account_balance,
                current_positions=current_positions,
                market_data=market_data
            )

            if not can_trade['can_trade']:
                logger.info(f"Cannot trade: {can_trade['reason']}")
                return

            # Calculate position size and risk parameters
            if decision['action'] in ['buy', 'sell']:
                position_params = self._calculate_trade_parameters(
                    decision, market_data, account_balance
                )
                
                # Execute trade
                if position_params:
                    await self._execute_trade(decision, position_params)

        except Exception as e:
            logger.error(f"Error executing trading logic: {e}")
            logger.exception("Full traceback:")

    async def _manage_positions(self, market_data: Dict[str, Any]) -> None:
        """Manage existing positions"""
        try:
            positions = self.position_manager.get_positions()
            for position in positions:
                # Prepare features for position analysis
                position_features = self._prepare_features(market_data)
                
                # Get ML model evaluation for position
                position_decision = await self._get_trading_decision(position_features)
                
                # Update position management parameters
                await self._update_position_parameters(
                    position, position_decision, market_data
                )
                
                # Check exit conditions
                if self._should_exit_position(position, position_decision, market_data):
                    await self._exit_position(position)

        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            logger.exception("Full traceback:")

    def _calculate_trade_parameters(self, decision: Dict[str, Any], 
                                  market_data: Dict[str, Any],
                                  account_balance: float) -> Optional[Dict[str, Any]]:
        """Calculate trade parameters based on ML decision"""
        try:
            current_price = market_data['historical_data']['close'].iloc[-1]
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                price=current_price,
                ml_confidence=decision['confidence'],
                available_margin=account_balance,
                market_data=market_data
            )

            # Calculate risk parameters
            risk_params = self.risk_manager.calculate_risk_parameters(
                entry_price=current_price,
                ml_confidence=decision['confidence'],
                market_data=market_data
            )

            return {
                'position_size': position_size,
                'entry_price': current_price,
                'stop_loss': risk_params['stop_loss'],
                'target': risk_params['target'],
                'trailing_stop': risk_params['trailing_stop']
            }

        except Exception as e:
            logger.error(f"Error calculating trade parameters: {e}")
            return None

    async def _execute_trade(self, decision: Dict[str, Any], 
                           trade_params: Dict[str, Any]) -> None:
        """Execute trade with given parameters"""
        try:
            order_type = 'BUY' if decision['action'] == 'buy' else 'SELL'
            
            order_id = await self.order_executor.place_order(
                symbol=Config.TRADING_SYMBOL,
                quantity=trade_params['position_size'],
                order_type=order_type,
                price=trade_params['entry_price']
            )

            if order_id:
                # Update position tracking
                self.position_manager.update_position(
                    symbol=Config.TRADING_SYMBOL,
                    quantity=trade_params['position_size'],
                    entry_price=trade_params['entry_price'],
                    stop_loss=trade_params['stop_loss'],
                    target=trade_params['target']
                )

                # Send notification
                self.notification_manager.send_trade_notification({
                    'type': order_type,
                    'symbol': Config.TRADING_SYMBOL,
                    'quantity': trade_params['position_size'],
                    'price': trade_params['entry_price'],
                    'confidence': decision['confidence']
                })

                logger.info(f"Trade executed successfully: {order_type}")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            logger.exception("Full traceback:")

    async def _update_position_parameters(self, position: Dict[str, Any],
                                        decision: Dict[str, Any],
                                        market_data: Dict[str, Any]) -> None:
        """Update position parameters based on new ML evaluation"""
        try:
            updated_params = self.risk_manager.calculate_dynamic_exits(
                position=position,
                current_price=market_data['historical_data']['close'].iloc[-1],
                ml_prediction=decision
            )

            if updated_params:
                self.position_manager.update_position_parameters(
                    symbol=position['symbol'],
                    parameters=updated_params
                )

        except Exception as e:
            logger.error(f"Error updating position parameters: {e}")

    def _should_exit_position(self, position: Dict[str, Any],
                            decision: Dict[str, Any],
                            market_data: Dict[str, Any]) -> bool:
        """Determine if position should be exited"""
        try:
            current_price = market_data['historical_data']['close'].iloc[-1]
            
            # Check stop loss and target
            if current_price <= position['stop_loss'] or current_price >= position['target']:
                return True

            # Check ML model confidence
            if decision['confidence'] < Config.ML_EXIT_THRESHOLD:
                return True

            # Check trailing stop
            if position.get('trailing_stop') and current_price <= position['trailing_stop']:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False

    async def _exit_position(self, position: Dict[str, Any]) -> None:
        """Exit given position"""
        try:
            await self.order_executor.exit_position(position['symbol'])
            self.position_manager.remove_position(position['symbol'])
            
            self.notification_manager.send_trade_notification({
                'type': 'EXIT',
                'symbol': position['symbol'],
                'quantity': position['quantity'],
                'exit_price': position['current_price']
            })

        except Exception as e:
            logger.error(f"Error exiting position: {e}")

    def _update_performance_metrics(self) -> None:
        """Update system performance metrics"""
        try:
            metrics = self.performance_tracker.get_performance_metrics()
            
            # Check if retraining is needed
            if self.performance_tracker.should_retrain():
                self._schedule_model_retraining()

            # Update stored metrics
            self.performance_metrics = metrics
            
            # Send performance update if needed
            if self._should_send_performance_update():
                self.notification_manager.send_performance_update(metrics)

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def stop_trading(self) -> None:
        """Stop trading system"""
        try:
            logger.info("Stopping trading system...")
            self.running = False
            
            # Exit all positions
            await self.position_manager.exit_all_positions()
            
            # Save final metrics
            self._save_final_metrics()
            
            # Send final notification
            self.notification_manager.send_alert(
                "system_shutdown",
                {"reason": "Trading system stopped", "time": datetime.now(ist)}
            )

        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")

    def _check_system_health(self) -> bool:
        """Check health of all system components"""
        try:
            checks = {
                'ml_model': self.ml_model is not None,
                'data_fetcher': self.data_fetcher.is_healthy(),
                'risk_manager': self.risk_manager is not None,
                'position_manager': self.position_manager is not None,
                'order_executor': self.order_executor is not None
            }
            
            return all(checks.values())

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return False

    async def _handle_error(self) -> None:
        """Handle system errors"""
        try:
            # Increment error counter
            self.error_count = getattr(self, 'error_count', 0) + 1
            
            if self.error_count >= Config.MAX_ERRORS:
                logger.critical("Maximum errors reached. Stopping trading system.")
                await self.stop_trading()
            else:
                # Wait before retrying
                await asyncio.sleep(Config.ERROR_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Error in error handler: {e}")

    async def _handle_trading_pause(self) -> None:
        """Handle trading system pause"""
        try:
            logger.info("Trading system paused")
            
            # Exit all positions if outside trading hours
            if not self._is_trading_hours():
                await self.position_manager.exit_all_positions()
            
            # Save current metrics
            self._save_metrics()
            
            # Wait before next check
            await asyncio.sleep(Config.PAUSE_CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error handling trading pause: {e}")

    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            margins = await self.kite.margins()
            return float(margins.get('equity', {}).get('available', {}).get('cash', 0.0))
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        current_time = datetime.now(ist).time()
        return Config.MARKET_OPEN_TIME <= current_time <= Config.MARKET_CLOSE_TIME

    def _schedule_model_retraining(self) -> None:
        """Schedule ML model retraining"""
        try:
            current_time = datetime.now(ist)
            
            # Check if enough time has passed since last retraining
            if (self.last_model_retrain and 
                (current_time - self.last_model_retrain).days < Config.RETRAIN_INTERVAL_DAYS):
                return

            logger.info("Scheduling model retraining")
            
            # Create retraining task
            asyncio.create_task(self._retrain_model())
            
        except Exception as e:
            logger.error(f"Error scheduling model retraining: {e}")

    async def _retrain_model(self) -> None:
        """Retrain ML model"""
        try:
            logger.info("Starting model retraining")
            
            # Fetch training data
            market_data = await self._update_market_data()
            if not market_data:
                raise Exception("Failed to fetch training data")

            # Prepare features for training
            features = self._prepare_features(market_data)
            
            # Retrain model
            if await self.ml_model.retrain(features):
                self.last_model_retrain = datetime.now(ist)
                logger.info("Model retraining completed successfully")
                
                # Update performance tracker
                self.performance_tracker.reset_metrics()
            else:
                logger.error("Model retraining failed")
                
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            logger.exception("Full traceback:")

    def _save_metrics(self) -> None:
        """Save current system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(ist).isoformat(),
                'performance': self.performance_metrics,
                'risk': self.risk_manager.get_risk_report(),
                'positions': self.position_manager.get_positions(),
                'ml_model': {
                    'last_retrain': self.last_model_retrain.isoformat() if self.last_model_retrain else None,
                    'confidence_metrics': self.performance_tracker.get_confidence_metrics()
                }
            }
            
            metrics_file = Path(Config.DATA_DIR) / 'system_metrics.json'
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _save_final_metrics(self) -> None:
        """Save final system metrics before shutdown"""
        try:
            final_metrics = {
                'end_time': datetime.now(ist).isoformat(),
                'total_trades': self.performance_tracker.get_total_trades(),
                'final_pnl': self.position_manager.get_total_pnl(),
                'win_rate': self.performance_tracker.get_win_rate(),
                'max_drawdown': self.risk_manager.get_max_drawdown(),
                'system_uptime': self._calculate_uptime()
            }
            
            metrics_file = Path(Config.DATA_DIR) / f'final_metrics_{datetime.now(ist).strftime("%Y%m%d")}.json'
            with open(metrics_file, 'w') as f:
                json.dump(final_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving final metrics: {e}")

    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        try:
            if hasattr(self, 'start_time'):
                uptime = datetime.now(ist) - self.start_time
                return str(uptime)
            return "Unknown"
        except Exception as e:
            logger.error(f"Error calculating uptime: {e}")
            return "Error"

    def _should_send_performance_update(self) -> bool:
        """Determine if performance update should be sent"""
        try:
            if not hasattr(self, 'last_performance_update'):
                self.last_performance_update = datetime.now(ist)
                return True
                
            time_since_update = (datetime.now(ist) - self.last_performance_update).seconds
            return time_since_update >= Config.PERFORMANCE_UPDATE_INTERVAL
            
        except Exception as e:
            logger.error(f"Error checking performance update timing: {e}")
            return False

    async def _monitor_system_resources(self) -> None:
        """Monitor system resource usage"""
        try:
            while self.running:
                # Monitor memory usage
                memory_usage = self._get_memory_usage()
                if memory_usage > Config.MAX_MEMORY_USAGE:
                    logger.warning(f"High memory usage detected: {memory_usage}%")
                    
                # Monitor CPU usage
                cpu_usage = self._get_cpu_usage()
                if cpu_usage > Config.MAX_CPU_USAGE:
                    logger.warning(f"High CPU usage detected: {cpu_usage}%")
                    
                await asyncio.sleep(Config.RESOURCE_CHECK_INTERVAL)
                
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")

    @staticmethod
    async def run_trading_system():
        """Run the trading system"""
        trader = None
        try:
            trader = BankNiftyTrader()
            trader.start_time = datetime.now(ist)
            
            # Start resource monitoring
            asyncio.create_task(trader._monitor_system_resources())
            
            # Start trading
            await trader.start_trading()
            
        except Exception as e:
            logger.critical(f"Critical error in trading system: {e}")
            logger.exception("Full traceback:")
            
        finally:
            if trader:
                await trader.stop_trading()

def main():
    """Main entry point"""
    try:
        # Set up event loop
        loop = asyncio.get_event_loop()
        
        # Run the trading system
        loop.run_until_complete(BankNiftyTrader.run_trading_system())
        
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        logger.exception("Full traceback:")
        
    finally:
        loop.close()

if __name__ == "__main__":
    main()