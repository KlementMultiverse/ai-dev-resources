"""
Utility functions
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> Dict:
    """Load JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return {}


def save_json(data: Dict, filepath: str):
    """Save data to JSON file"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise


def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def format_size(bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"
