from __future__ import annotations

from sklearn.svm import SVC

from app.core.config import RANDOM_STATE
from app.plugins.classification_base import ClassificationPlugin
from app.plugins.registry import register_classification_plugin


@register_classification_plugin
class SVMClassifierPlugin(ClassificationPlugin):
    name = "svm"

    def param_grid(self, config: dict) -> list[dict]:
        return [{"C": c} for c in config.get("C", [1.0])]

    def build_model(self, params: dict):
        # probability=True enables predict_proba (needed for ROC-AUC/log loss) at the cost
        # of an internal cross-validation pass during fit — acceptable for the dataset
        # sizes this platform targets; linear kernel keeps that cost down further.
        return SVC(kernel="linear", C=params["C"], probability=True, random_state=RANDOM_STATE)
