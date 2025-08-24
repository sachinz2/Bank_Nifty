"""
Encapsulates the machine learning model for generating trading signals.
"""
import joblib
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLSignalGenerator:
    def __init__(self, model_path: Path):
        self.model = None
        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"ML model loaded successfully from {model_path}")
            else:
                logger.error(f"ML model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")

    def get_signal(self, features: pd.DataFrame) -> str:
        """Generates a trading signal ('BUY', 'SELL', 'HOLD') from features."""
        if self.model is None:
            return "HOLD"
        
        try:
            prediction = self.model.predict(features.tail(1))
            return str(prediction[0]).upper()
        except Exception as e:
            logger.error(f"Error during ML model prediction: {e}")
            return "HOLD"