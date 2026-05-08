"""
config_writer.py
────────────────
Serialises the classified column assignments into `rec_config.json`.
Fully data-driven version supporting dynamic roles.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

SCHEMA_VERSION = "1.0"


def _to_native(obj):
    """
    Recursively convert non-native types (numpy.float32, etc.) to standard
    Python types so they can be JSON serialized.
    """
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        try:
            if "." in str(obj):
                return float(obj)
            return int(obj)
        except (ValueError, TypeError):
            return str(obj)


class LockFileExistsError(RuntimeError):
    """Raised when rec_config.lock exists."""


class ConfigWriter:
    """
    Writes rec_config.json and manages the lock file.
    """

    def __init__(self, output_dir: str):
        self._output_dir  = output_dir
        self._config_path = os.path.join(output_dir, "rec_config.json")
        self._lock_path   = os.path.join(output_dir, "rec_config.lock")

    def check_lock(self) -> None:
        """Raise LockFileExistsError if Feature Engineering holds the lock."""
        if os.path.exists(self._lock_path):
            with open(self._lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            raise LockFileExistsError(
                f"rec_config.lock exists — Feature Engineering (PID "
                f"{lock_data.get('pid', '?')}) is currently reading "
                f"rec_config.json."
            )

    def build_config(
        self,
        role_buckets: Dict[str, List[str]],
        interaction_signals: Dict[str, float],
        ambiguous: List[dict],
        provenance: Dict[str, dict],
        weighted_role: str = "interaction_signals"
    ) -> dict:
        """
        Assemble the configuration dictionary dynamically.
        """
        config: Dict[str, Any] = {
            "schema_version":     SCHEMA_VERSION,
            "generated_at":       datetime.now(timezone.utc).isoformat(),
        }
        
        # Add all role buckets dynamically (except the weighted one which is special)
        for role, columns in role_buckets.items():
            if role != weighted_role:
                config[role] = sorted(set(columns))
        
        # Add weighted role (dictionary structure)
        config[weighted_role] = interaction_signals
        
        # Add metadata
        config["ambiguous"]          = ambiguous
        config["mapping_provenance"] = provenance
        
        return _to_native(config)

    def write(self, config: dict) -> str:
        """Write the configuration dictionary to rec_config.json."""
        os.makedirs(self._output_dir, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
        return self._config_path

    def load_existing(self) -> Optional[dict]:
        """Load an existing rec_config.json if present."""
        if not os.path.exists(self._config_path):
            return None
        with open(self._config_path, "r", encoding="utf-8") as f:
            return json.load(f)
