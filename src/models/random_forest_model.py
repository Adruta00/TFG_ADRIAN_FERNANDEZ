"""
random_forest_model.py
======================
Implementación de BaseModel usando RandomForestClassifier de scikit-learn.

Por qué Random Forest como primer modelo ML:
    - Robusto frente al sobreajuste (gracias al ensemble de árboles).
    - No requiere normalización estricta (aunque la recibe del pipeline).
    - Proporciona importancia de features de forma nativa.
    - Entrena rápido sobre miles de muestras (importante para el TFG).
    - Sirve como punto de comparación natural antes de LSTM (Hito 3).

Arquitectura de clases:
    RandomForestModel → hereda BaseModel → recibido por MLStrategy
    El BacktestEngine nunca toca este archivo.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Modelo Random Forest para clasificación de señales de trading.

    Mapeo de clases:
         1  →  BUY  (precio sube > threshold en target_horizon velas)
        -1  →  SELL (precio baja > threshold en target_horizon velas)
         0  →  HOLD (movimiento menor al threshold)

    Parameters
    ----------
    n_estimators : int
        Número de árboles del ensemble. Más = mejor pero más lento.
        Default: 100 (buen balance velocidad/calidad para el TFG).
    max_depth : int | None
        Profundidad máxima de cada árbol. None = sin límite (puede sobreajustar).
        Default: 10 (regulariza el modelo).
    min_samples_leaf : int
        Mínimo de muestras por hoja. Aumentar regulariza contra ruido.
    random_state : int
        Semilla para reproducibilidad (importante para el TFG).
    class_weight : str | None
        'balanced' compensa la distribución desigual de BUY/SELL/HOLD.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ) -> None:
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.class_weight     = class_weight

        self._model: RandomForestClassifier | None = None
        self._is_fitted = False
        self.feature_importances_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Interfaz BaseModel
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entrena el RandomForestClassifier.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Features normalizadas del conjunto de entrenamiento.
        y_train : np.ndarray, shape (n_samples,)
            Targets: 1, -1, 0.
        """
        logger.info(
            "Entrenando RandomForestClassifier | n_estimators=%d | max_depth=%s | "
            "n_samples=%d | n_features=%d",
            self.n_estimators, self.max_depth, len(X_train), X_train.shape[1],
        )

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=-1,   # Usa todos los núcleos disponibles
        )

        self._model.fit(X_train, y_train)
        self._is_fitted = True
        self.feature_importances_ = self._model.feature_importances_

        # Log de distribución de clases predichas en train (diagnóstico rápido)
        train_preds = self._model.predict(X_train)
        unique, counts = np.unique(train_preds, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(
            "Entrenamiento completado | Distribución en train: %s", class_dist
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones de clase para nuevas muestras.

        Returns
        -------
        np.ndarray de enteros: 1 (BUY), -1 (SELL), 0 (HOLD).
        """
        self._check_fitted()
        predictions = self._model.predict(X)
        return predictions.astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve probabilidades por clase (útil para análisis de confianza).

        Returns
        -------
        np.ndarray de shape (n_samples, n_classes).
        El orden de clases está en self._model.classes_.
        """
        self._check_fitted()
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Serializa el modelo entrenado con joblib.

        Parameters
        ----------
        path : str  Ruta del archivo (convención: terminar en .joblib).
        """
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":              self._model,
            "n_estimators":       self.n_estimators,
            "max_depth":          self.max_depth,
            "min_samples_leaf":   self.min_samples_leaf,
            "random_state":       self.random_state,
            "class_weight":       self.class_weight,
            "feature_importances": self.feature_importances_,
            "model_type":         "RandomForestModel",
        }
        joblib.dump(payload, path)
        logger.info("RandomForestModel guardado → %s", path)

    def load(self, path: str) -> None:
        """
        Carga un modelo previamente serializado.

        Parameters
        ----------
        path : str  Ruta del archivo creado por save().
        """
        payload = joblib.load(path)
        self._model               = payload["model"]
        self.n_estimators         = payload["n_estimators"]
        self.max_depth            = payload["max_depth"]
        self.min_samples_leaf     = payload["min_samples_leaf"]
        self.random_state         = payload["random_state"]
        self.class_weight         = payload["class_weight"]
        self.feature_importances_ = payload.get("feature_importances")
        self._is_fitted           = True
        logger.info("RandomForestModel cargado desde %s", path)

    # ------------------------------------------------------------------
    # Utilidades de análisis (para la memoria del TFG)
    # ------------------------------------------------------------------

    def get_feature_importance_report(
        self, feature_cols: list[str], window_size: int, top_n: int = 20
    ) -> list[tuple[str, float]]:
        """
        Devuelve los top_n features más importantes con nombres legibles.

        Las features tienen nombres compuestos:
            {nombre_indicador}_t-{timestep_relativo}
        Ejemplo: 'rsi_t-0' = RSI del último período de la ventana.

        Returns
        -------
        Lista de tuplas (nombre_feature, importancia) ordenadas descendentemente.
        """
        self._check_fitted()

        n_features = len(feature_cols)
        names = []
        for step in range(window_size):
            for col in feature_cols:
                offset = window_size - 1 - step
                names.append(f"{col}_t-{offset}")

        importances = self.feature_importances_
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a fit() antes de predict() o save()."
            )
