import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pytz
import time
import asyncio
import aiohttp
from kiteconnect import KiteConnect
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import backoff
from requests.exceptions import RequestException, Timeout
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
import json
from pathlib import Path
from ta import momentum

ist = pytz.timezone('Asia/Kolkata')

class DataFetcher:
    def __init__(self, api_key, access_token, kite):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = kite
        self.data_cache = {}
        self.cache_timestamps = {}
        self.banknifty_token = None
        self.token_bucket = TokenBucket(tokens=Config.API_RATE_LIMIT, 
                                      fill_rate=Config.API_RATE_LIMIT/Config.API_RATE_LIMIT_PERIOD)
        self._initialize_cache()

    @classmethod
    async def create(cls, api_key, access_token):
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        instance = cls(api_key, access_token, kite)
        instance.banknifty_token = await instance.get_banknifty_token()
        return instance

    def _initialize_cache(self):
        """Initialize data cache with required categories"""
        self.data_cache = {
            'historical_data': None,
            'options_chain': None,
            'market_depth': None,
            'sentiment_data': None,
            'volatility_data': None,
            'market_breadth': None
        }
        self.cache_timestamps = {k: None for k in self.data_cache.keys()}

    async def get_banknifty_token(self, exchange='NFO'):
        """Get instrument token for Bank Nifty"""
        try:
            instruments = await asyncio.to_thread(self.kite.instruments, exchange)
            for instrument in instruments:
                if instrument['tradingsymbol'] == 'BANKNIFTY':
                    return instrument['instrument_token']
            return None
        except Exception as e:
            logger.error(f"Error fetching banknifty token: {e}")
            return None

    @backoff.on_exception(backoff.expo,
                         (RequestException, Timeout),
                         max_tries=5,
                         max_time=300)
    async def fetch_all_data(self):
        """Fetch all required data in parallel"""
        try:
            logger.info("Fetching all market data...")
            
            # Create tasks for all data fetching operations
            tasks = [
                self.fetch_historical_data(),
                self.fetch_options_chain_data(),
                self.fetch_market_depth_data(),
                self.fetch_sentiment_data(),
                self.fetch_volatility_data(),
                self.fetch_market_breadth_data()
            ]
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            data = {
                'historical_data': results[0],
                'options_chain': results[1],
                'market_depth': results[2],
                'sentiment_data': results[3],
                'volatility_data': results[4],
                'market_breadth': results[5]
            }
            
            self._update_cache(data)
            return data

        except Exception as e:
            logger.error(f"Error fetching all data: {e}")
            logger.exception("Full traceback:")
            return None

    @backoff.on_exception(backoff.expo,
                         (RequestException, Timeout),
                         max_tries=5,
                         max_time=300)
    async def fetch_historical_data(self, interval='5minute', days=5):
        """Fetch historical price data with error handling and caching"""
        try:
            cache_key = 'historical_data'
            if self._is_cache_valid(cache_key, max_age_seconds=300):  # 5 minutes cache
                return self.data_cache[cache_key]

            end_date = datetime.now(ist)
            start_date = end_date - timedelta(days=days)
            
            data = await self._fetch_kite_historical(
                instrument_token=self.banknifty_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval
            )
            
            if data is None or data.empty:
                logger.error("Failed to fetch historical data")
                return self.data_cache.get(cache_key)  # Return cached data if available

            # Process the data
            data = await asyncio.to_thread(self._process_historical_data, data)
            
            # Update cache
            self.data_cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now(ist)
            
            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            logger.exception("Full traceback:")
            return self.data_cache.get(cache_key)

    @backoff.on_exception(backoff.expo,
                         (RequestException, Timeout),
                         max_tries=5,
                         max_time=300)
    async def fetch_options_chain_data(self):
        """Fetch and process options chain data"""
        try:
            cache_key = 'options_chain'
            if self._is_cache_valid(cache_key, max_age_seconds=60):  # 1 minute cache
                return self.data_cache[cache_key]

            # Get current BankNifty price
            spot_price = await self._fetch_spot_price()
            if not spot_price:
                return self.data_cache.get(cache_key)

            # Fetch options around spot price
            options_data = await self._fetch_options_around_price(spot_price)
            if not options_data:
                return self.data_cache.get(cache_key)

            # Process options data
            processed_data = await asyncio.to_thread(self._process_options_data, options_data, spot_price)
            
            # Calculate additional metrics
            processed_data.update({
                'put_call_ratio': await asyncio.to_thread(self._calculate_put_call_ratio, options_data),
                'iv_skew': await asyncio.to_thread(self._calculate_iv_skew, options_data, spot_price),
                'max_pain': await asyncio.to_thread(self._calculate_max_pain, options_data),
                'options_sentiment': await asyncio.to_thread(self._analyze_options_sentiment, options_data)
            })

            # Update cache
            self.data_cache[cache_key] = processed_data
            self.cache_timestamps[cache_key] = datetime.now(ist)
            
            return processed_data

        except Exception as e:
            logger.error(f"Error fetching options chain data: {e}")
            logger.exception("Full traceback:")
            return self.data_cache.get(cache_key)

    async def fetch_market_depth_data(self):
        """Fetch market depth data for BankNifty"""
        try:
            cache_key = 'market_depth'
            if self._is_cache_valid(cache_key, max_age_seconds=30):  # 30 seconds cache
                return self.data_cache[cache_key]

            market_depth = await self._fetch_kite_depth(self.banknifty_token)
            
            if market_depth:
                processed_depth = await asyncio.to_thread(self._process_market_depth, market_depth)
                
                # Update cache
                self.data_cache[cache_key] = processed_depth
                self.cache_timestamps[cache_key] = datetime.now(ist)
                
                return processed_depth

            return self.data_cache.get(cache_key)

        except Exception as e:
            logger.error(f"Error fetching market depth: {e}")
            logger.exception("Full traceback:")
            return self.data_cache.get(cache_key)

    async def fetch_sentiment_data(self):
        """Fetch and analyze market sentiment data"""
        try:
            cache_key = 'sentiment_data'
            if self._is_cache_valid(cache_key, max_age_seconds=300):  # 5 minutes cache
                return self.data_cache[cache_key]

            # Fetch various sentiment indicators
            news_sentiment = await self._fetch_news_sentiment()
            social_sentiment = await self._fetch_social_sentiment()
            market_mood = await self._analyze_market_mood()
            
            sentiment_data = {
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'market_mood': market_mood,
                'combined_sentiment': await asyncio.to_thread(self._calculate_combined_sentiment,
                    news_sentiment, social_sentiment, market_mood
                )
            }

            # Update cache
            self.data_cache[cache_key] = sentiment_data
            self.cache_timestamps[cache_key] = datetime.now(ist)
            
            return sentiment_data

        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            logger.exception("Full traceback:")
            return self.data_cache.get(cache_key)

    async def fetch_volatility_data(self):
        """Fetch and calculate volatility metrics"""
        try:
            cache_key = 'volatility_data'
            if self._is_cache_valid(cache_key, max_age_seconds=300):  # 5 minutes cache
                return self.data_cache[cache_key]

            # Calculate historical volatility
            historical_data = await self.fetch_historical_data()
            hist_volatility = await asyncio.to_thread(self._calculate_historical_volatility, historical_data)

            # Get implied volatility from options
            options_data = await self.fetch_options_chain_data()
            implied_volatility = await asyncio.to_thread(self._calculate_implied_volatility, options_data)

            volatility_data = {
                'historical_volatility': hist_volatility,
                'implied_volatility': implied_volatility,
                'volatility_spread': implied_volatility - hist_volatility,
                'volatility_trend': await asyncio.to_thread(self._analyze_volatility_trend, historical_data)
            }

            # Update cache
            self.data_cache[cache_key] = volatility_data
            self.cache_timestamps[cache_key] = datetime.now(ist)
            
            return volatility_data

        except Exception as e:
            logger.error(f"Error fetching volatility data: {e}")
            logger.exception("Full traceback:")
            return self.data_cache.get(cache_key)

    async def fetch_market_breadth_data(self):
        """Fetch market breadth indicators"""
        try:
            cache_key = 'market_breadth'
            if self._is_cache_valid(cache_key, max_age_seconds=300):  # 5 minutes cache
                return self.data_cache[cache_key]

            # Fetch data for market breadth calculation
            bank_nifty_stocks = await self._fetch_bank_nifty_stocks()
            
            advancing = 0
            declining = 0
            total_volume = 0
            
            tasks = [self._fetch_stock_data(stock) for stock in bank_nifty_stocks]
            stock_data_list = await asyncio.gather(*tasks)

            for stock_data in stock_data_list:
                if stock_data['close'] > stock_data['prev_close']:
                    advancing += 1
                else:
                    declining += 1
                total_volume += stock_data['volume']

            breadth_data = {
                'advance_decline_ratio': advancing / declining if declining > 0 else float('inf'),
                'market_breadth': (advancing - declining) / (advancing + declining),
                'volume_trend': await asyncio.to_thread(self._analyze_volume_trend, total_volume),
                'market_strength': await asyncio.to_thread(self._calculate_market_strength, bank_nifty_stocks)
            }

            # Update cache
            self.data_cache[cache_key] = breadth_data
            self.cache_timestamps[cache_key] = datetime.now(ist)
            
            return breadth_data

        except Exception as e:
            logger.error(f"Error fetching market breadth data: {e}")
            logger.exception("Full traceback:")
            return self.data_cache.get(cache_key)

    # Helper methods for data processing and analysis
    def _process_historical_data(self, data):
        """Process historical price data"""
        try:
            # Add basic technical indicators
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['SMA50'] = data['close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            data['VWAP'] = self._calculate_vwap(data)
            
            return data
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            return data

    def _process_options_data(self, options_data, spot_price):
        """Process options chain data"""
        try:
            processed = {
                'calls': [],
                'puts': [],
                'atm_strike': self._find_atm_strike(options_data, spot_price)
            }
            
            for option in options_data:
                option_type = 'calls' if option['instrument_type'] == 'CE' else 'puts'
                processed[option_type].append({
                    'strike': option['strike'],
                    'ltp': option['last_price'],
                    'iv': option['implied_volatility'],
                    'delta': option['delta'],
                    'gamma': option['gamma'],
                    'theta': option['theta'],
                    'vega': option['vega'],
                    'oi': option['open_interest'],
                    'volume': option['volume']
                })
            
            return processed
        except Exception as e:
            logger.error(f"Error processing options data: {e}")
            return None

    def _is_cache_valid(self, key, max_age_seconds):
        """Check if cached data is still valid"""
        if key not in self.cache_timestamps or self.cache_timestamps[key] is None:
            return False
            
        age = (datetime.now(ist) - self.cache_timestamps[key]).total_seconds()
        return age < max_age_seconds

    def _update_cache(self, data):
        """Update cache with new data"""
        try:
            current_time = datetime.now(ist)
            for key, value in data.items():
                if value is not None:
                    self.data_cache[key] = value
                    self.cache_timestamps[key] = current_time
        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    # Additional helper methods for calculations and analysis
    def _calculate_historical_volatility(self, data, window=20):
        """Calculate historical volatility"""
        try:
            if data is None or data.empty:
                return None
                
            returns = np.log(data['close'] / data['close'].shift(1))
            return returns.std() * np.sqrt(252) * 100
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return None

    def _calculate_implied_volatility(self, options_data):
        """Calculate average implied volatility"""
        try:
            if not options_data or 'atm_strike' not in options_data:
                return None
                
            atm_strike = options_data['atm_strike']
            atm_options = [opt for opt in options_data['calls'] + options_data['puts'] 
                         if opt['strike'] == atm_strike]
                         
            if not atm_options:
                return None
                
            return sum(opt['iv'] for opt in atm_options) / len(atm_options)
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return None

    def _analyze_volume_trend(self, current_volume, window=20):
        """Analyze volume trend"""
        try:
            historical_data = self.data_cache.get('historical_data')
            if historical_data is None:
                return 'neutral'
                
            avg_volume = historical_data['volume'].rolling(window=window).mean().iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                return 'high'
            elif current_volume < avg_volume * 0.5:
                return 'low'
            return 'normal'
            
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {e}")
            return 'neutral'

    async def _analyze_market_mood(self):
        """Analyze overall market mood using multiple indicators"""
        try:
            indicators = {
                'price_trend': await asyncio.to_thread(self._analyze_price_trend),
                'volume_trend': await asyncio.to_thread(self._analyze_volume_trend, self.data_cache.get('historical_data', {}).get('volume', 0)),
                'options_sentiment': await asyncio.to_thread(self._analyze_options_sentiment, self.data_cache.get('options_chain')),
                'market_breadth': await asyncio.to_thread(self._analyze_market_breadth)
            }
            
            # Calculate mood score
            mood_score = 0
            weights = {'price_trend': 0.4, 'volume_trend': 0.2, 
                      'options_sentiment': 0.2, 'market_breadth': 0.2}
            
            for indicator, value in indicators.items():
                if value == 'bullish':
                    mood_score += weights[indicator]
                elif value == 'bearish':
                    mood_score -= weights[indicator]
            
            if mood_score > 0.2:
                return 'bullish'
            elif mood_score < -0.2:
                return 'bearish'
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error analyzing market mood: {e}")
            return 'neutral'

    def _analyze_price_trend(self, window=20):
        """Analyze price trend"""
        try:
            data = self.data_cache.get('historical_data')
            if data is None or data.empty:
                return 'neutral'
            
            current_price = data['close'].iloc[-1]
            sma = data['close'].rolling(window=window).mean().iloc[-1]
            std_dev = data['close'].rolling(window=window).std().iloc[-1]
            
            if current_price > sma + std_dev:
                return 'bullish'
            elif current_price < sma - std_dev:
                return 'bearish'
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error analyzing price trend: {e}")
            return 'neutral'

    def _analyze_volatility_trend(self, data):
        """Analyze volatility trend"""
        try:
            if data is None or data.empty:
                return 'normal'
            
            current_volatility = self._calculate_historical_volatility(data.tail(20))
            past_volatility = self._calculate_historical_volatility(data.tail(40).head(20))
            
            if current_volatility > past_volatility * 1.2:
                return 'increasing'
            elif current_volatility < past_volatility * 0.8:
                return 'decreasing'
            return 'stable'
            
        except Exception as e:
            logger.error(f"Error analyzing volatility trend: {e}")
            return 'normal'

    @sleep_and_retry
    @limits(calls=Config.API_RATE_LIMIT, period=Config.API_RATE_LIMIT_PERIOD)
    async def _fetch_kite_historical(self, instrument_token, from_date, to_date, interval):
        """Fetch historical data from Kite"""
        try:
            data = await asyncio.to_thread(
                self.kite.historical_data,
                instrument_token,
                from_date,
                to_date,
                interval
            )
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching Kite historical data: {e}")
            return None

    async def _fetch_kite_depth(self, instrument_token):
        """Fetch market depth from Kite"""
        try:
            depth = await asyncio.to_thread(self.kite.depth, instrument_token)
            return depth
        except Exception as e:
            logger.error(f"Error fetching Kite market depth: {e}")
            return None

    async def _fetch_spot_price(self):
        """Fetch spot price for BankNifty"""
        try:
            quote = await asyncio.to_thread(self.kite.ltp, f"NFO:BANKNIFTY")
            return quote[f"NFO:BANKNIFTY"]['last_price']
        except Exception as e:
            logger.error(f"Error fetching spot price: {e}")
            return None

    async def _fetch_bank_nifty_stocks(self):
        """Fetch list of stocks in Bank Nifty"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(Config.BANK_NIFTY_CONSTITUENTS_URL) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('constituents', [])
            return []
        except Exception as e:
            logger.error(f"Error fetching Bank Nifty stocks: {e}")
            return []

    async def _fetch_options_around_price(self, spot_price):
        """Fetch options data around current spot price"""
        try:
            strike_step = 100
            strikes_range = 5  # Number of strikes above and below spot
            
            base_strike = round(spot_price / strike_step) * strike_step
            strikes = range(
                base_strike - strike_step * strikes_range,
                base_strike + strike_step * (strikes_range + 1),
                strike_step
            )
            
            options_data = []
            tasks = []
            
            async with aiohttp.ClientSession() as session:
                for strike in strikes:
                    for option_type in ['CE', 'PE']:
                        tasks.append(self._fetch_option_data(session, strike, option_type))
                
                options_data = await asyncio.gather(*tasks)
            
            return [data for data in options_data if data is not None]
            
        except Exception as e:
            logger.error(f"Error fetching options around price: {e}")
            return []

    async def _fetch_option_data(self, session, strike, option_type):
        # This is a dummy implementation. You need to replace it with actual API calls to fetch option data.
        return {
            'instrument_type': option_type,
            'strike': strike,
            'last_price': 100,
            'implied_volatility': 0.2,
            'delta': 0.5,
            'gamma': 0.1,
            'theta': -0.1,
            'vega': 0.2,
            'open_interest': 1000,
            'volume': 100
        }


    async def _fetch_stock_data(self, stock_symbol):
        # This is a dummy implementation. You need to replace it with actual API calls to fetch stock data.
        return {
            'close': 100,
            'prev_close': 99,
            'volume': 10000
        }


    async def _fetch_news_sentiment(self):
        """Fetch and analyze news sentiment"""
        try:
            # Implement news API integration here
            # For now, returning dummy sentiment
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'confidence': 0.5
            }
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return None

    async def _fetch_social_sentiment(self):
        """Fetch and analyze social media sentiment"""
        try:
            # Implement social media sentiment analysis here
            # For now, returning dummy sentiment
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'volume': 0
            }
        except Exception as e:
            logger.error(f"Error fetching social sentiment: {e}")
            return None

    def _calculate_combined_sentiment(self, news_sentiment, social_sentiment, market_mood):
        """Calculate combined sentiment score"""
        try:
            weights = {
                'news': 0.3,
                'social': 0.2,
                'market': 0.5
            }
            
            components = {
                'news': self._normalize_sentiment(news_sentiment.get('sentiment_score', 0)),
                'social': self._normalize_sentiment(social_sentiment.get('sentiment_score', 0)),
                'market': 1 if market_mood == 'bullish' else -1 if market_mood == 'bearish' else 0
            }
            
            combined_score = sum(score * weights[component] 
                               for component, score in components.items())
            
            return {
                'score': combined_score,
                'label': 'bullish' if combined_score > 0.2 else 'bearish' if combined_score < -0.2 else 'neutral',
                'components': components
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined sentiment: {e}")
            return {'score': 0, 'label': 'neutral', 'components': {}}

    def _normalize_sentiment(self, score):
        """Normalize sentiment score to [-1, 1] range"""
        try:
            return max(min(score, 1), -1)
        except Exception as e:
            logger.error(f"Error normalizing sentiment: {e}")
            return 0

    def save_market_snapshot(self):
        """Save current market data snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now(ist).isoformat(),
                'spot_price': self.data_cache.get('spot_price'),
                'market_mood': self._analyze_market_mood(),
                'volatility': self.data_cache.get('volatility_data'),
                'sentiment': self.data_cache.get('sentiment_data'),
                'market_breadth': self.data_cache.get('market_breadth')
            }
            
            snapshot_path = Path(Config.DATA_DIR) / 'market_snapshots'
            snapshot_path.mkdir(parents=True, exist_ok=True)
            
            file_path = snapshot_path / f"snapshot_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}.json"
            with open(file_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
            
            logger.info(f"Market snapshot saved: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving market snapshot: {e}")

    def get_market_status(self):
        """Get current market status summary"""
        try:
            return {
                'timestamp': datetime.now(ist),
                'data_freshness': {
                    key: self._get_data_age(key) 
                    for key in self.cache_timestamps.keys()
                },
                'market_mood': self._analyze_market_mood(),
                'data_quality': self._assess_data_quality()
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return None

    def _get_data_age(self, key):
        """Get age of cached data"""
        if self.cache_timestamps.get(key) is None:
            return None
        return (datetime.now(ist) - self.cache_timestamps[key]).total_seconds()

    def _assess_data_quality(self):
        """Assess quality of cached data"""
        try:
            quality_scores = {}
            for key, data in self.data_cache.items():
                if data is None:
                    quality_scores[key] = 0
                    continue
                
                age = self._get_data_age(key)
                completeness = 1.0  # Implement completeness check
                reliability = 1.0  # Implement reliability check
                
                quality_scores[key] = {
                    'age_score': max(0, 1 - age/3600) if age else 0,  # Decay over 1 hour
                    'completeness': completeness,
                    'reliability': reliability,
                    'overall_score': (max(0, 1 - age/3600) if age else 0 + completeness + reliability) / 3
                }
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {}

    def _calculate_put_call_ratio(self, options_data):
        """
        Calculate Put-Call Ratio
        options_data: list of option data dictionaries
        """
        call_volume = sum(opt['volume'] for opt in options_data if opt['instrument_type'] == 'CE')
        put_volume = sum(opt['volume'] for opt in options_data if opt['instrument_type'] == 'PE')
        
        if call_volume == 0:
            return float('inf')  # Avoid division by zero
        
        return put_volume / call_volume

    def _calculate_iv_skew(self, options_data, spot_price):
        # This is a dummy implementation. You need to provide the logic for IV skew calculation.
        logger.warning("IV Skew calculation is a dummy implementation.")
        return 0.0

    def _calculate_max_pain(self, options_data):
        # This is a dummy implementation. You need to provide the logic for max pain calculation.
        logger.warning("Max pain calculation is a dummy implementation.")
        return 30000

    def _analyze_options_sentiment(self, options_data):
        # This is a dummy implementation. You need to provide the logic for options sentiment analysis.
        logger.warning("Options sentiment analysis is a dummy implementation.")
        return 'neutral'

    def _process_market_depth(self, market_depth):
        # This is a dummy implementation. You need to provide the logic for processing market depth.
        logger.warning("Market depth processing is a dummy implementation.")
        return market_depth

    def _calculate_market_strength(self, bank_nifty_stocks):
        # This is a dummy implementation. You need to provide the logic for market strength calculation.
        logger.warning("Market strength calculation is a dummy implementation.")
        return 0.5

    def _find_atm_strike(self, options_data, spot_price):
        # This is a dummy implementation. You need to provide the logic for finding the ATM strike.
        logger.warning("ATM strike finding is a dummy implementation.")
        return round(spot_price / 100) * 100

    def _calculate_rsi(self, close_prices, window=14):
        return momentum.RSIIndicator(close_prices, window=window).rsi().iloc[-1]

    def _calculate_vwap(self, data):
        return (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum().iloc[-1]
    
    def _analyze_market_breadth(self, stock_data_list):
        advancing = sum(1 for stock in stock_data_list if stock['close'] > stock['close'])
        declining = sum(1 for stock in stock_data_list if stock['close'] < stock['close'])

        if advancing > declining:
            return 'bullish'
        elif declining > advancing:
            return 'bearish'
        else:
            return 'neutral'


class TokenBucket:
    def __init__(self, tokens, fill_rate):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.timestamp = time.time()

    async def consume(self, tokens):
        await self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def refill(self):
        now = time.time()
        delta = now - self.timestamp
        self.tokens = min(self.capacity, self.tokens + delta * self.fill_rate)
        self.timestamp = now

if __name__ == "__main__":
    # Add test code here
    pass