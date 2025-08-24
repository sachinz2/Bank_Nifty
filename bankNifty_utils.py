import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import json
from pathlib import Path

ist = pytz.timezone('Asia/Kolkata')

def is_market_open():
    """Check if market is currently open"""
    current_time = datetime.now(ist).time()
    return Config.MARKET_OPEN_TIME <= current_time <= Config.MARKET_CLOSE_TIME

def get_time_to_market_close():
    """Get remaining time until market close"""
    current_time = datetime.now(ist)
    close_time = datetime.combine(current_time.date(), Config.MARKET_CLOSE_TIME)
    return (close_time - current_time).total_seconds() / 60  # in minutes

def format_number(number, decimals=2):
    """Format number with appropriate decimals"""
    try:
        return round(float(number), decimals)
    except (ValueError, TypeError):
        return 0.0

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values"""
    try:
        return ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
    except (ValueError, TypeError):
        return 0.0

def save_to_json(data, filename):
    """Save data to JSON file"""
    try:
        filepath = Path(Config.DATA_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder)
        return True
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        return False

def load_from_json(filename):
    """Load data from JSON file"""
    try:
        filepath = Path(Config.DATA_DIR) / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading from JSON: {e}")
        return None

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

def calculate_lot_size(quantity):
    """Calculate appropriate lot size"""
    return (quantity // Config.LOT_SIZE) * Config.LOT_SIZE

def is_high_impact_news_time():
    """Check if current time is around high-impact news events"""
    current_time = datetime.now(ist).time()
    # Add your specific news event times here
    news_times = [
        (time(9, 15), time(9, 30)),  # Market opening
        (time(15, 15), time(15, 30))  # Market closing
    ]
    return any(start <= current_time <= end for start, end in news_times)

def validate_price(price):
    """Validate if price is reasonable"""
    return isinstance(price, (int, float)) and price > 0

def calculate_risk_reward_ratio(entry_price, stop_loss, target):
    """Calculate risk-reward ratio"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        return reward / risk if risk != 0 else 0
    except Exception:
        return 0

def log_trade_details(trade_data):
    """Log trade details to file"""
    try:
        filepath = Path(Config.DATA_DIR) / 'trade_log.json'
        existing_trades = []
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing_trades = json.load(f)
        
        existing_trades.append(trade_data)
        
        with open(filepath, 'w') as f:
            json.dump(existing_trades, f, indent=2, cls=DateTimeEncoder)
    except Exception as e:
        logger.error(f"Error logging trade details: {e}")

def get_expiry_type(expiry_date):
    """Determine if expiry is weekly or monthly"""
    expiry = pd.Timestamp(expiry_date)
    last_day = pd.Timestamp(expiry.year, expiry.month, 1) + pd.offsets.MonthEnd(1)
    return 'monthly' if expiry.date() == last_day.date() else 'weekly'

def is_valid_quantity(quantity):
    """Check if quantity is valid"""
    return isinstance(quantity, int) and quantity > 0 and quantity <= Config.MAX_POSITION_QUANTITY

def calculate_breakeven_price(entry_price, quantity, transaction_cost):
    """Calculate breakeven price including transaction costs"""
    total_cost = (entry_price * quantity) + transaction_cost
    return total_cost / quantity

# Add more utility functions as needed