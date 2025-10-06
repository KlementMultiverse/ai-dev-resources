"""
Configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    def __init__(self):
        self.env = os.getenv('ENV', 'development')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return os.getenv(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'env': self.env,
            'debug': self.debug,
            'log_level': self.log_level
        }


# Global config instance
config = Config()
