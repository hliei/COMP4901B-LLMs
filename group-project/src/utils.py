"""
Simple utility to load environment variables from .env file.
"""

import os
from pathlib import Path


def load_env(env_path: str = None):
    """Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file (default: .env in project root)
    """
    if env_path is None:
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                os.environ[key] = value


# Auto-load .env when module is imported
load_env()
