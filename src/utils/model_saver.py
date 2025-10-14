"""
Model saving and loading utilities for the exploration phase.
Implements logic to save best models per classifier and skip training if model already exists.
"""

import os
import pickle
import joblib
from pathlib import Path
from typing import Optional, Any, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSaver:
    """
    Utility class for saving and loading models during exploration phase.

    Logic:
    - If best model exists (is saved), skip this classifier
    - Else, if no model exists:
        - Execute RandomizedSearchCV
        - Save model
    """

    def __init__(self, base_dir: str = "src/models/exploration_phase"):
        """
        Initialize ModelSaver with base directory for model storage.

        Args:
            base_dir: Base directory for saving models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def model_exists(
        self, classifier_name: str, experiment_name: str = "default"
    ) -> bool:
        """
        Check if a model for the given classifier and experiment already exists.

        Args:
            classifier_name: Name of the classifier (e.g., 'LogisticRegression', 'RandomForest')
            experiment_name: Name of the experiment (e.g., 'no_sampling', 'with_sampling')

        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = self._get_model_path(classifier_name, experiment_name)
        return model_path.exists()

    def save_model(
        self,
        classifier_name: str,
        model: Any,
        experiment_name: str = "default",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save a trained model with optional metadata.

        Args:
            classifier_name: Name of the classifier
            model: The trained model object
            experiment_name: Name of the experiment
            metadata: Optional metadata to save with the model

        Returns:
            str: Path to the saved model file
        """
        model_path = self._get_model_path(classifier_name, experiment_name)

        # Save the model
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        # Save metadata if provided
        if metadata:
            metadata_path = self._get_metadata_path(classifier_name, experiment_name)
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            logger.info(f"Metadata saved: {metadata_path}")

        return str(model_path)

    def load_model(self, classifier_name: str, experiment_name: str = "default") -> Any:
        """
        Load a saved model.

        Args:
            classifier_name: Name of the classifier
            experiment_name: Name of the experiment

        Returns:
            The loaded model object

        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = self._get_model_path(classifier_name, experiment_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")
        return model

    def load_metadata(
        self, classifier_name: str, experiment_name: str = "default"
    ) -> Optional[Dict]:
        """
        Load metadata for a saved model.

        Args:
            classifier_name: Name of the classifier
            experiment_name: Name of the experiment

        Returns:
            Dict with metadata or None if no metadata exists
        """
        metadata_path = self._get_metadata_path(classifier_name, experiment_name)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        logger.info(f"Metadata loaded: {metadata_path}")
        return metadata

    def get_model_info(
        self, classifier_name: str, experiment_name: str = "default"
    ) -> Dict:
        """
        Get information about a saved model.

        Args:
            classifier_name: Name of the classifier
            experiment_name: Name of the experiment

        Returns:
            Dict with model information
        """
        model_path = self._get_model_path(classifier_name, experiment_name)
        metadata_path = self._get_metadata_path(classifier_name, experiment_name)

        info = {
            "exists": model_path.exists(),
            "model_path": str(model_path),
            "metadata_exists": metadata_path.exists(),
            "metadata_path": str(metadata_path),
        }

        if model_path.exists():
            stat = model_path.stat()
            info.update({"size_bytes": stat.st_size, "modified_time": stat.st_mtime})

        return info

    def list_saved_models(self) -> Dict[str, Dict]:
        """
        List all saved models and their information.

        Returns:
            Dict with model information
        """
        models_info = {}

        for model_file in self.base_dir.glob("*.joblib"):
            # Extract classifier and experiment from filename
            # Format: {classifier_name}_{experiment_name}.joblib
            name_parts = model_file.stem.split("_")
            if len(name_parts) >= 2:
                experiment_name = name_parts[-1]
                classifier_name = "_".join(name_parts[:-1])
            else:
                classifier_name = name_parts[0]
                experiment_name = "default"

            key = f"{classifier_name}_{experiment_name}"
            models_info[key] = self.get_model_info(classifier_name, experiment_name)

        return models_info

    def _get_model_path(self, classifier_name: str, experiment_name: str) -> Path:
        """Get the path for a model file."""
        filename = f"{classifier_name}_{experiment_name}.joblib"
        return self.base_dir / filename

    def _get_metadata_path(self, classifier_name: str, experiment_name: str) -> Path:
        """Get the path for a metadata file."""
        filename = f"{classifier_name}_{experiment_name}_metadata.pkl"
        return self.base_dir / filename


def create_model_saver(base_dir: str = "src/models/exploration_phase") -> ModelSaver:
    """
    Factory function to create a ModelSaver instance.

    Args:
        base_dir: Base directory for saving models

    Returns:
        ModelSaver instance
    """
    return ModelSaver(base_dir)
