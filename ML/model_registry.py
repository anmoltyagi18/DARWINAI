# =============================================================================
# MODULE: ML/model_registry.py
# PROJECT: AIGOFIN - AI Quant Trading Platform
#
# PURPOSE:
#   Central store for all AI models used across the platform.
#   Provides save/load, version control, metadata tracking, and a clean
#   interface for other modules to retrieve the latest trained model.
#
# FOLDER STRUCTURE:
#   models/
#     rl_models/          ← Reinforcement learning agents (rl_trader.py)
#     strategy_models/    ← Strategy classifiers (strategy_evolver.py, ai_brain.py)
#
# INTEGRATES WITH:
#   rl_trader.py, strategy_evolver.py, ai_brain.py
#
# AUTHOR: AIGOFIN System
# =============================================================================

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ModelCategory = Literal["rl_models", "strategy_models"]

CATEGORY_DIRS: Dict[str, str] = {
    "rl_models": "rl_models",
    "strategy_models": "strategy_models",
}

METADATA_FILENAME = "registry.json"


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Unified model management for all AI components in AIGOFIN.

    Responsibilities
    ----------------
    * Register a model with metadata (name, version, category, metrics).
    * Persist models to disk using joblib (compatible with sklearn, stable
      baselines, and custom objects).
    * Load any version or the latest version of a named model.
    * Maintain a JSON registry file per category for fast listing.

    Directory layout
    ----------------
    <base_dir>/
      rl_models/
        registry.json
        ppo_agent_v1.joblib
        ppo_agent_v2.joblib
      strategy_models/
        registry.json
        xgb_classifier_v1.joblib

    Usage
    -----
    registry = ModelRegistry(base_dir="models")
    registry.save_model(model, name="ppo_agent", category="rl_models",
                        metrics={"sharpe": 1.45})
    agent = registry.load_model(name="ppo_agent", category="rl_models")
    """

    def __init__(self, base_dir: str = "models") -> None:
        """
        Parameters
        ----------
        base_dir : str
            Root directory for model storage. Created if it does not exist.
        """
        self.base_dir = Path(base_dir)
        self._init_directories()
        logger.info(f"ModelRegistry initialised at: {self.base_dir.resolve()}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_directories(self) -> None:
        """Create category sub-directories and registry JSON files."""
        for category, subdir in CATEGORY_DIRS.items():
            cat_path = self.base_dir / subdir
            cat_path.mkdir(parents=True, exist_ok=True)

            registry_path = cat_path / METADATA_FILENAME
            if not registry_path.exists():
                self._write_registry(category, {})
                logger.debug(f"Created registry: {registry_path}")

    # ------------------------------------------------------------------
    # Registry I/O helpers
    # ------------------------------------------------------------------

    def _registry_path(self, category: ModelCategory) -> Path:
        return self.base_dir / CATEGORY_DIRS[category] / METADATA_FILENAME

    def _read_registry(self, category: ModelCategory) -> Dict[str, Any]:
        """Load the category registry dict from disk."""
        path = self._registry_path(category)
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_registry(
        self, category: ModelCategory, data: Dict[str, Any]
    ) -> None:
        """Persist the category registry dict to disk."""
        path = self._registry_path(category)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def _next_version(
        self, name: str, category: ModelCategory
    ) -> int:
        """Return the next sequential version number for a named model."""
        registry = self._read_registry(category)
        versions = [
            entry["version"]
            for entry in registry.get(name, {}).get("versions", [])
        ]
        return max(versions, default=0) + 1

    def _model_filename(self, name: str, version: int) -> str:
        return f"{name}_v{version}.joblib"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        category: ModelCategory,
        version: int,
        filepath: str,
        metrics: Optional[Dict[str, float]] = None,
        description: str = "",
    ) -> None:
        """
        Register model metadata in the category registry without saving
        the object itself. Use save_model() to both save and register.

        Parameters
        ----------
        name : str
            Logical model name (e.g. 'ppo_agent', 'xgb_strategy').
        category : ModelCategory
            'rl_models' or 'strategy_models'.
        version : int
            Version integer.
        filepath : str
            Path where the model file is stored.
        metrics : dict, optional
            Performance metrics to store (e.g. sharpe, win_rate).
        description : str
            Human-readable description.
        """
        self._validate_category(category)
        registry = self._read_registry(category)

        entry = {
            "version": version,
            "filepath": str(filepath),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
            "description": description,
        }

        if name not in registry:
            registry[name] = {"versions": []}
        registry[name]["versions"].append(entry)

        self._write_registry(category, registry)
        logger.info(
            f"Registered [{category}] {name} v{version} — "
            f"metrics: {metrics}"
        )

    def save_model(
        self,
        model: Any,
        name: str,
        category: ModelCategory,
        metrics: Optional[Dict[str, float]] = None,
        description: str = "",
    ) -> str:
        """
        Serialise model to disk and register it.

        Parameters
        ----------
        model : Any
            The model object (sklearn estimator, SB3 agent, etc.).
        name : str
            Logical model name.
        category : ModelCategory
            'rl_models' or 'strategy_models'.
        metrics : dict, optional
            Performance metrics snapshot at save time.
        description : str
            Optional description.

        Returns
        -------
        str
            Absolute path to saved model file.
        """
        self._validate_category(category)

        version = self._next_version(name, category)
        filename = self._model_filename(name, version)
        filepath = self.base_dir / CATEGORY_DIRS[category] / filename

        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")

        self.register_model(
            name=name,
            category=category,
            version=version,
            filepath=str(filepath),
            metrics=metrics,
            description=description,
        )
        return str(filepath)

    def load_model(
        self,
        name: str,
        category: ModelCategory,
        version: Optional[int] = None,
    ) -> Any:
        """
        Load a model from disk by name and optional version.

        Parameters
        ----------
        name : str
            Logical model name.
        category : ModelCategory
            'rl_models' or 'strategy_models'.
        version : int, optional
            Specific version to load. If None, loads the latest version.

        Returns
        -------
        Any
            Deserialised model object.

        Raises
        ------
        FileNotFoundError
            If no model with the given name/version exists.
        """
        self._validate_category(category)

        if version is None:
            return self.get_latest_model(name, category)

        registry = self._read_registry(category)
        if name not in registry:
            raise FileNotFoundError(
                f"No model named '{name}' found in category '{category}'."
            )

        versions = registry[name]["versions"]
        match = next((v for v in versions if v["version"] == version), None)
        if match is None:
            available = [v["version"] for v in versions]
            raise FileNotFoundError(
                f"Version {version} not found for '{name}'. "
                f"Available: {available}"
            )

        filepath = Path(match["filepath"])
        if not filepath.exists():
            raise FileNotFoundError(
                f"Model file missing on disk: {filepath}"
            )

        model = joblib.load(filepath)
        logger.info(f"Loaded [{category}] {name} v{version} from {filepath}")
        return model

    def get_latest_model(
        self, name: str, category: ModelCategory
    ) -> Any:
        """
        Load the highest-versioned model for a given name.

        Parameters
        ----------
        name : str
            Logical model name.
        category : ModelCategory
            'rl_models' or 'strategy_models'.

        Returns
        -------
        Any
            Deserialised model object.
        """
        self._validate_category(category)
        registry = self._read_registry(category)

        if name not in registry or not registry[name]["versions"]:
            raise FileNotFoundError(
                f"No versions registered for model '{name}' in '{category}'."
            )

        # Sort by version descending, pick the highest
        versions = sorted(
            registry[name]["versions"],
            key=lambda v: v["version"],
            reverse=True,
        )
        latest = versions[0]
        filepath = Path(latest["filepath"])

        if not filepath.exists():
            raise FileNotFoundError(
                f"Latest model file missing: {filepath}"
            )

        model = joblib.load(filepath)
        logger.info(
            f"Loaded latest [{category}] {name} v{latest['version']} "
            f"from {filepath}"
        )
        return model

    def list_models(
        self, category: Optional[ModelCategory] = None
    ) -> Dict[str, Any]:
        """
        List all registered models.

        Parameters
        ----------
        category : ModelCategory, optional
            If given, list only models in that category.
            If None, list all categories.

        Returns
        -------
        dict
            Nested dict: { category: { name: { versions: [...] } } }
        """
        categories = (
            [category]
            if category
            else list(CATEGORY_DIRS.keys())
        )
        result = {}
        for cat in categories:
            result[cat] = self._read_registry(cat)

        return result

    def delete_model(
        self,
        name: str,
        category: ModelCategory,
        version: Optional[int] = None,
    ) -> None:
        """
        Remove a model version from disk and registry.
        If version is None, removes ALL versions of the model.

        Parameters
        ----------
        name : str
            Logical model name.
        category : ModelCategory
            Target category.
        version : int, optional
            Specific version. None = delete all.
        """
        self._validate_category(category)
        registry = self._read_registry(category)

        if name not in registry:
            logger.warning(f"delete_model: '{name}' not found in '{category}'.")
            return

        versions = registry[name]["versions"]

        if version is None:
            # Remove all versions
            for v in versions:
                self._remove_file(v["filepath"])
            del registry[name]
            logger.info(f"Deleted all versions of [{category}] '{name}'.")
        else:
            # Remove specific version
            to_remove = [v for v in versions if v["version"] == version]
            for v in to_remove:
                self._remove_file(v["filepath"])
            registry[name]["versions"] = [
                v for v in versions if v["version"] != version
            ]
            if not registry[name]["versions"]:
                del registry[name]
            logger.info(f"Deleted [{category}] '{name}' v{version}.")

        self._write_registry(category, registry)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_category(category: str) -> None:
        if category not in CATEGORY_DIRS:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid options: {list(CATEGORY_DIRS.keys())}"
            )

    @staticmethod
    def _remove_file(filepath: str) -> None:
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted file: {path}")
        else:
            logger.warning(f"File not found for deletion: {path}")
