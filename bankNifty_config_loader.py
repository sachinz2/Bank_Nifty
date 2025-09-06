import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(file_path: Path):
    """
    Loads a YAML configuration file.

    Args:
        file_path: The path to the YAML file.

    Returns:
        A dictionary with the configuration, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {file_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file at {file_path}: {e}")
        return None
