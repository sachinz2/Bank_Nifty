    pytz
from datetime import time
import os

class Config:
    # API and Authentication
    KITE_API_KEY = os.environ.get('KITE_API_KEY')
    KITE_API_SECRET = os.environ.get('KITE_API_SECRET')

    # Trading Parameters
    MAX_OPEN_TRADES = 1
    MAX_RISK_PER_TRADE = 6000  # Maximum risk per trade in INR
    MAX_RISK_PERCENTAGE = 0.03  # Maximum risk as a percentage of available balance
    DAILY_PROFIT_LIMIT = 100000  # Daily profit limit in INR
    MAX_DAILY_LOSS = 15000  # Maximum daily loss limit in INR
    FIXED_QUANTITY = 15  # Minimum quantity to trade
    LOT_SIZE = 15
    MAX_POSITION_SIZE = 0.5  # Maximum position size as a fraction of available balance
    
    # Risk Management
    MAX_RISK_PER_TRADE_PERCENT = 1  # Maximum risk per trade as a percentage of available balance
    MAX_LOSS_PERCENT = 6  # Maximum loss percentage before exiting a position
    PROFIT_THRESHOLD_PERCENT = 10  # Profit percentage to start monitoring for exit
    MAX_DAILY_LOSS_PERCENT = 3  # Maximum daily loss as a percentage of total investment
    DAILY_PROFIT_LIMIT_PERCENT = 5  # Daily profit limit as a percentage of total investment
    BANKNIFTY_INSTRUMENT_TOKEN = None  # This will be set dynamically
    GLOBAL_EXPIRY = None
    MIN_INVESTMENT = 100000

    # Time and Schedule
    MARKET_OPEN_TIME = time(9, 15)
    MARKET_CLOSE_TIME = time(15, 30)
    AVOID_TRADING_START = time(15, 00)
    AVOID_TRADING_END = time(15, 15)
    TRADE_INTERVAL = 3  # Minimum time between trades in minutes

    # Strategy Parameters
    HISTORICAL_DATA_DAYS = 60
    ML_WEIGHT = 1.5
    ML_TRAINING_DAYS = 100
    ML_RETRAIN_INTERVAL = 7  # Retrain ML model every 7 days
    OPTIONS_WEIGHT = 1.5  # Weight for options analysis
    TREND_WEIGHT = 2.0  # Weight for market trend analysis
    ORIGINAL_BUY_SIGNAL_THRESHOLD = 0.45
    ORIGINAL_SELL_SIGNAL_THRESHOLD = 0.35
    BUY_SIGNAL_THRESHOLD = ORIGINAL_BUY_SIGNAL_THRESHOLD
    SELL_SIGNAL_THRESHOLD = ORIGINAL_SELL_SIGNAL_THRESHOLD
    ADX_THRESHOLD = 25
    VOLATILE_ATR_MULTIPLIER = 1.5

    # ML Model Parameters
    ML_BUY_THRESHOLD = 0.55
    ML_SELL_THRESHOLD = 0.40
    STOP_LOSS_PERCENT = 4.0
    TAKE_PROFIT_PERCENT = 13.0
    TRAILING_STOP_ACTIVATION_PERCENT = 10.0
    TRAILING_STOP_DISTANCE_PERCENT = 0.5

    # Technical Analysis
    SMA_FAST = 20
    SMA_SLOW = 50
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    # Minimum Data Points
    MIN_DATA_POINTS = {
        'ML_MODEL': 1000,  # Minimum data points required for ML model training
        'TECHNICAL_ANALYSIS': 50,  # Minimum data points for technical analysis
        'SENTIMENT_ANALYSIS': 100  # Minimum data points for sentiment analysis
    }

    # XGBoost parameters
    XGB_N_ESTIMATORS = 100
    XGB_LEARNING_RATE = 0.1
    XGB_MAX_DEPTH = 5

    # Retraining parameters
    RETRAIN_INTERVAL_DAYS = 7  # Retrain every week

    # Dynamic risk management
    BASE_STOP_LOSS_PERCENT = 5.0
    BASE_TAKE_PROFIT_PERCENT = 10.0

    # Other Parameters
    LOG_LEVEL = 'INFO'
    DATA_DIR = 'data'
    MODEL_DIR = 'models'

    # API Rate Limiting
    API_RATE_LIMIT = 3  # Number of API calls allowed per second
    API_RATE_LIMIT_PERIOD = 1  # Time period for rate limit in seconds

    # Volatility Thresholds
    HIGH_VOLATILITY_THRESHOLD = 0.4
    LOW_VOLATILITY_THRESHOLD = 0.2

    # Position Sizing
    MAX_POSITION_QUANTITY = 900  # Maximum allowed quantity for a single position

    # Fibonacci Levels
    FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

    # Ichimoku Cloud Parameters
    ICHIMOKU_CONVERSION_LINE_PERIOD = 9
    ICHIMOKU_BASE_LINE_PERIOD = 26
    ICHIMOKU_LEADING_SPAN_B_PERIOD = 52
    ICHIMOKU_DISPLACEMENT = 26

    # Volume Profile
    VOLUME_PROFILE_LEVELS = 10

    # Pivot Points
    USE_PIVOT_POINTS = True

    # Chart Patterns
    CHART_PATTERN_LOOKBACK = 20

    # Candlestick Patterns
    USE_CANDLESTICK_PATTERNS = True

    # Market Breadth
    USE_MARKET_BREADTH = True

    # Option Greeks
    RISK_FREE_RATE = 0.05  # Risk-free rate for option Greeks calculation

    # Performance Monitoring
    PERFORMANCE_EVALUATION_INTERVAL = 100  # Number of trades after which to evaluate performance

    # Backtesting
    BACKTEST_START_DATE = '2022-01-01'
    BACKTEST_END_DATE = '2023-12-31'

    # Logging
    LOG_FILE = 'banknifty_trading.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Notifications
    ENABLE_EMAIL_NOTIFICATIONS = False
    EMAIL_RECIPIENT = 'your_email@example.com'

    # Data Storage
    USE_DATABASE = False
    DATABASE_URL = 'sqlite:///banknifty_trading.db'

    # Enhanced ML Model Parameters
    ML_PREDICTION_THRESHOLD = 0.65
    ML_CONFIDENCE_MINIMUM = 0.55
    ML_RETRAIN_HOURS = 24
    ML_FEATURE_ENGINEERING = {
        'use_technical': True,
        'use_sentiment': True,
        'use_options': True,
        'use_market_breadth': True
    }
    
# Keep this line to maintain compatibility with existing scripts
config = Config()