import asyncio
from datetime import datetime
import pytz
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
from kiteconnect_common import get_kite
import backoff
from requests.exceptions import RequestException, Timeout
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

ist = pytz.timezone('Asia/Kolkata')

class OrderExecutor:
    def __init__(self):
        self.kite = get_kite()
        self.pending_orders = {}
        self.executed_orders = {}
        self.order_history = []
        self.last_order_time = None
        self.order_count = 0
        self.order_lock = asyncio.Lock()
        self._load_order_history()

    async def place_order(self, 
                         symbol: str, 
                         quantity: int, 
                         order_type: str, 
                         price: float = None,
                         sl_price: float = None,
                         target_price: float = None,
                         ml_confidence: float = None) -> Optional[str]:
        """Place order with ML-based parameters"""
        try:
            async with self.order_lock:
                if not await self._can_place_order():
                    logger.warning("Order placement not allowed at this time")
                    return None

                # Validate order parameters
                if not self._validate_order_params(symbol, quantity, price):
                    return None

                # Prepare order parameters
                order_params = await self._prepare_order_params(
                    symbol, quantity, order_type, price, sl_price, target_price
                )

                # Place the order
                order_id = await self._execute_order(order_params)
                if not order_id:
                    return None

                # Track the order
                await self._track_order(order_id, order_params, ml_confidence)

                logger.info(f"Order placed successfully: {order_id}")
                return order_id

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            logger.exception("Full traceback:")
            return None

    @backoff.on_exception(backoff.expo,
                         (RequestException, Timeout),
                         max_tries=5,
                         max_time=300)
    async def _execute_order(self, order_params: Dict[str, Any]) -> Optional[str]:
        """Execute the order with retry mechanism"""
        try:
            order_id = await self.kite.place_order(**order_params)
            
            # Initial order verification
            if not await self._verify_order_placement(order_id):
                logger.error(f"Order verification failed for order_id: {order_id}")
                return None
                
            return order_id

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None

    async def modify_order(self, 
                          order_id: str, 
                          new_price: float = None,
                          new_quantity: int = None,
                          new_sl_price: float = None,
                          new_target_price: float = None) -> bool:
        """Modify existing order"""
        try:
            async with self.order_lock:
                if order_id not in self.pending_orders:
                    logger.error(f"Order {order_id} not found in pending orders")
                    return False

                modify_params = {
                    'order_id': order_id,
                    'variety': self.kite.VARIETY_REGULAR
                }

                if new_price:
                    modify_params['price'] = new_price
                if new_quantity:
                    modify_params['quantity'] = new_quantity
                if new_sl_price:
                    modify_params['trigger_price'] = new_sl_price
                if new_target_price:
                    modify_params['squareoff'] = new_target_price

                await self.kite.modify_order(**modify_params)
                
                # Update order tracking
                self.pending_orders[order_id].update({
                    'modified_time': datetime.now(ist),
                    'modification_count': self.pending_orders[order_id].get('modification_count', 0) + 1
                })

                return True

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            async with self.order_lock:
                await self.kite.cancel_order(
                    variety=self.kite.VARIETY_REGULAR,
                    order_id=order_id
                )

                if order_id in self.pending_orders:
                    # Move to order history
                    self.order_history.append({
                        **self.pending_orders[order_id],
                        'status': 'CANCELLED',
                        'cancel_time': datetime.now(ist).isoformat()
                    })
                    del self.pending_orders[order_id]

                return True

        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def place_bracket_order(self,
                                symbol: str,
                                quantity: int,
                                entry_price: float,
                                stop_loss: float,
                                target: float,
                                ml_confidence: float) -> Optional[str]:
        """Place bracket order with ML-based parameters"""
        try:
            async with self.order_lock:
                if not await self._can_place_order():
                    return None

                order_params = {
                    'variety': self.kite.VARIETY_BO,
                    'exchange': self.kite.EXCHANGE_NFO,
                    'tradingsymbol': symbol,
                    'transaction_type': self.kite.TRANSACTION_TYPE_BUY,
                    'quantity': quantity,
                    'price': entry_price,
                    'squareoff': abs(target - entry_price),
                    'stoploss': abs(entry_price - stop_loss),
                    'trailing_stoploss': self._calculate_trailing_sl(ml_confidence, entry_price, stop_loss)
                }

                order_id = await self.kite.place_order(**order_params)
                if order_id:
                    await self._track_order(order_id, order_params, ml_confidence)
                    return order_id

                return None

        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            return None

    async def _track_order(self, 
                          order_id: str, 
                          order_params: Dict[str, Any],
                          ml_confidence: float) -> None:
        """Track order details"""
        try:
            self.pending_orders[order_id] = {
                'order_params': order_params,
                'status': 'PENDING',
                'place_time': datetime.now(ist).isoformat(),
                'ml_confidence': ml_confidence,
                'modifications': [],
                'fills': []
            }

            # Start order monitoring
            asyncio.create_task(self._monitor_order(order_id))

        except Exception as e:
            logger.error(f"Error tracking order {order_id}: {e}")

    async def _monitor_order(self, order_id: str) -> None:
        """Monitor order status and updates"""
        try:
            while order_id in self.pending_orders:
                order_info = await self.kite.order_history(order_id)
                if not order_info:
                    await asyncio.sleep(1)
                    continue

                latest_status = order_info[-1]['status']
                
                if latest_status == 'COMPLETE':
                    await self._handle_order_completion(order_id, order_info[-1])
                    break
                elif latest_status in ['REJECTED', 'CANCELLED']:
                    await self._handle_order_rejection(order_id, order_info[-1])
                    break

                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}")

    async def _handle_order_completion(self, 
                                     order_id: str, 
                                     order_info: Dict[str, Any]) -> None:
        """Handle completed order"""
        try:
            async with self.order_lock:
                if order_id not in self.pending_orders:
                    return

                order_data = self.pending_orders.pop(order_id)
                order_data.update({
                    'completion_time': datetime.now(ist).isoformat(),
                    'execution_price': order_info['average_price'],
                    'filled_quantity': order_info['filled_quantity'],
                    'status': 'COMPLETE'
                })

                self.executed_orders[order_id] = order_data
                self.order_history.append(order_data)

                await self._save_order_history()

        except Exception as e:
            logger.error(f"Error handling order completion for {order_id}: {e}")

    async def _handle_order_rejection(self, 
                                    order_id: str, 
                                    order_info: Dict[str, Any]) -> None:
        """Handle rejected order"""
        try:
            async with self.order_lock:
                if order_id not in self.pending_orders:
                    return

                order_data = self.pending_orders.pop(order_id)
                order_data.update({
                    'rejection_time': datetime.now(ist).isoformat(),
                    'rejection_reason': order_info.get('status_message', 'Unknown'),
                    'status': order_info['status']
                })

                self.order_history.append(order_data)
                await self._save_order_history()

        except Exception as e:
            logger.error(f"Error handling order rejection for {order_id}: {e}")

    def _calculate_trailing_sl(self, 
                             ml_confidence: float, 
                             entry_price: float, 
                             initial_sl: float) -> float:
        """Calculate trailing stop loss based on ML confidence"""
        try:
            base_trail = abs(entry_price - initial_sl)
            
            # Adjust trailing stop based on ML confidence
            if ml_confidence >= 0.8:
                return base_trail * 0.8  # Tighter trailing for high confidence
            elif ml_confidence >= 0.6:
                return base_trail
            else:
                return base_trail * 1.2  # Wider trailing for low confidence

        except Exception as e:
            logger.error(f"Error calculating trailing stop loss: {e}")
            return base_trail

    async def _verify_order_placement(self, order_id: str) -> bool:
        """Verify order placement was successful"""
        try:
            max_retries = 5
            retry_delay = 1

            for _ in range(max_retries):
                order_info = await self.kite.order_history(order_id)
                if order_info:
                    status = order_info[-1]['status']
                    if status not in ['REJECTED', 'CANCELLED']:
                        return True
                    return False

                await asyncio.sleep(retry_delay)

            return False

        except Exception as e:
            logger.error(f"Error verifying order placement: {e}")
            return False

    async def _can_place_order(self) -> bool:
        """Check if order can be placed"""
        try:
            current_time = datetime.now(ist)

            # Check trading hours
            if not self._is_trading_hours(current_time):
                return False

            # Check order frequency
            if self.last_order_time and \
               (current_time - self.last_order_time).seconds < Config.MIN_ORDER_INTERVAL:
                return False

            # Check daily order limit
            if self.order_count >= Config.MAX_ORDERS_PER_DAY:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking order placement conditions: {e}")
            return False

    def _validate_order_params(self, 
                             symbol: str, 
                             quantity: int, 
                             price: float) -> bool:
        """Validate order parameters"""
        try:
            if not symbol or not isinstance(symbol, str):
                logger.error("Invalid symbol")
                return False

            if not quantity or quantity <= 0:
                logger.error("Invalid quantity")
                return False

            if price and price <= 0:
                logger.error("Invalid price")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating order parameters: {e}")
            return False

    async def _prepare_order_params(self,
                                  symbol: str,
                                  quantity: int,
                                  order_type: str,
                                  price: float,
                                  sl_price: float,
                                  target_price: float) -> Dict[str, Any]:
        """Prepare order parameters"""
        params = {
            'variety': self.kite.VARIETY_REGULAR,
            'exchange': self.kite.EXCHANGE_NFO,
            'tradingsymbol': symbol,
            'transaction_type': order_type,
            'quantity': quantity,
            'product': self.kite.PRODUCT_MIS,
            'order_type': self.kite.ORDER_TYPE_MARKET if price is None else self.kite.ORDER_TYPE_LIMIT
        }

        if price:
            params['price'] = price
        if sl_price:
            params['trigger_price'] = sl_price
        if target_price:
            params['squareoff'] = target_price

        return params

    def _is_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours"""
        return Config.MARKET_OPEN_TIME <= current_time.time() <= Config.MARKET_CLOSE_TIME

    async def _save_order_history(self) -> None:
        """Save order history to file"""
        try:
            history_file = Path(Config.DATA_DIR) / 'order_history.json'
            history_file.parent.mkdir(parents=True, exist_ok=True)

            with open(history_file, 'w') as f:
                json.dump(self.order_history, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving order history: {e}")

    def _load_order_history(self) -> None:
        """Load order history from file"""
        try:
            history_file = Path(Config.DATA_DIR) / 'order_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.order_history = json.load(f)

        except Exception as e:
            logger.error(f"Error loading order history: {e}")

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current order status"""
        if order_id in self.executed_orders:
            return self.executed_orders[order_id]
        elif order_id in self.pending_orders:
            return self.pending_orders[order_id]
        return {"status": "NOT_FOUND"}

    async def get_order_history(self) -> List[Dict[str, Any]]:
        """Get complete order history"""
        return self.order_history

    def reset_daily_counters(self) -> None:
        """Reset daily order counters"""
        self.order_count = 0
        self.last_order_time = None

if __name__ == "__main__":
    # Test code
    pass