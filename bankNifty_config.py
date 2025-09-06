class Config:
    # Feature Engineering
    MAX_FEATURES = 50
    USE_PCA = False

    # Strategy
    ML_WEIGHT = 0.5
    BUY_SIGNAL_THRESHOLD = 0.6
    SELL_SIGNAL_THRESHOLD = 0.6

    # Data Fetcher
    API_RATE_LIMIT = 10
    API_RATE_LIMIT_PERIOD = 1 # in seconds
    BANK_NIFTY_CONSTITUENTS_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK"
    DATA_DIR = "data"

    # Volatility
    HIGH_VOLATILITY_THRESHOLD = 0.4
    LOW_VOLATILITY_THRESHOLD = 0.2
