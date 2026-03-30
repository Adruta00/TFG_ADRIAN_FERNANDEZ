"""
baseline_model.py
=================
Implementación de BaseModel que replica la lógica del SMA crossover del Hito 1.

Por qué existe este archivo:
    - Permite comparar modelos ML contra el benchmark rule-based usando exactamente
      el mismo pipeline (mismas features, mismo backtesting, mismas métricas).
    - Si se comparara MLStrategy contra RuleBasedStrategy directamente, habría
      diferencias de implementación que contaminarían la comparación.
    - Con BaselineModel, ambos pasan por MLStrategy y BacktestEngine de la misma forma.

Lógica interna:
    La feature matrix contiene sma_fast y sma_slow como columnas.
    El modelo aplica la misma lógica de cruce de medias pero sobre las
    features del período más reciente de la ventana (último timestep).

    Dado que el StandardScaler preserva el orden relativo (solo escala linealmente),
    la comparación sma_fast > sma_slow sigue siendo válida tras la normalización.

IMPORTANTE: fit() no hace nada (no hay parámetros que aprender).
            El modelo es determinista y no requiere datos de entrenamiento.
"""

import logging
from pathlib import Path

import joblib
import numpy as np

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BaselineModel(BaseModel):
    """
    Baseline rule-based: SMA crossover envuelto en la interfaz BaseModel.

    Parameters
    ----------
    feature_cols : list[str]
        Lista de columnas de features en el mismo orden en que aparecen en X.
        Necesario para localizar los índices de sma_fast y sma_slow.
    window_size : int
        Tamaño de la ventana deslizante usada en create_feature_matrix.
        Necesario para extraer los valores del último timestep de la ventana.
    """

    def __init__(self, feature_cols: list[str], window_size: int) -> None:
        self.feature_cols = feature_cols
        self.window_size  = window_size
        self._is_fitted   = False

        # Índices de sma_fast y sma_slow en el vector de features por timestep
        self._idx_fast: int | None = None
        self._idx_slow: int | None = None

        if "sma_fast" in feature_cols:
            self._idx_fast = feature_cols.index("sma_fast")
        if "sma_slow" in feature_cols:
            self._idx_slow = feature_cols.index("sma_slow")

        if self._idx_fast is None or self._idx_slow is None:
            logger.warning(
                "BaselineModel: 'sma_fast' o 'sma_slow' no encontradas en feature_cols. "
                "El modelo devolverá HOLD=0 para todas las predicciones."
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        No entrena nada: la lógica es puramente determinista.
        Se llama para mantener la interfaz uniforme con otros modelos.
        """
        self._is_fitted = True
        logger.info("BaselineModel.fit() completado (sin parámetros que aprender).")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica la lógica SMA crossover sobre las features del último timestep
        de cada ventana.

        Cómo funciona el indexado:
            X.shape = (n_samples, window_size * n_features)
            El último timestep empieza en índice: (window_size - 1) * n_features
            sma_fast en el último timestep: (window_size-1)*n_features + idx_fast
            sma_slow en el último timestep: (window_size-1)*n_features + idx_slow

            La comparación sma_fast > sma_slow funciona incluso con valores
            normalizados porque StandardScaler preserva el orden relativo.

        Returns
        -------
        np.ndarray con valores 1 (BUY), -1 (SELL), 0 (HOLD).
        Nota: Este modelo solo genera BUY o HOLD (sin SELL), igual que la
        estrategia original del Hito 1 que solo cierra posiciones en cruce bajista.
        """
        n_samples  = X.shape[0]
        n_features = len(self.feature_cols)
        predictions = np.zeros(n_samples, dtype=np.int32)

        if self._idx_fast is None or self._idx_slow is None:
            return predictions  # Todo HOLD

        # Índices de sma_fast y sma_slow en el ÚLTIMO timestep de la ventana
        last_ts_offset = (self.window_size - 1) * n_features
        col_fast = last_ts_offset + self._idx_fast
        col_slow = last_ts_offset + self._idx_slow

        # Valores del timestep ANTERIOR (penúltimo) para detectar el cruce
        prev_ts_offset = (self.window_size - 2) * n_features
        prev_fast = X[:, prev_ts_offset + self._idx_fast]
        prev_slow = X[:, prev_ts_offset + self._idx_slow]
        curr_fast = X[:, col_fast]
        curr_slow = X[:, col_slow]

        # Cruce alcista: fast cruza POR ENCIMA de slow → BUY
        bullish = (curr_fast > curr_slow) & (prev_fast <= prev_slow)
        # Cruce bajista: fast cruza POR DEBAJO de slow → SELL
        bearish = (curr_fast < curr_slow) & (prev_fast >= prev_slow)

        predictions[bullish] = 1    # BUY
        predictions[bearish] = -1   # SELL
        # El resto permanece 0 (HOLD)

        buy_count  = bullish.sum()
        sell_count = bearish.sum()
        logger.debug("BaselineModel: %d BUY, %d SELL, %d HOLD",
                     buy_count, sell_count, n_samples - buy_count - sell_count)
        return predictions

    def save(self, path: str) -> None:
        """Serializa la configuración del modelo."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_cols": self.feature_cols,
            "window_size":  self.window_size,
            "model_type":   "BaselineModel",
        }
        joblib.dump(payload, path)
        logger.info("BaselineModel guardado → %s", path)

    def load(self, path: str) -> None:
        """Carga la configuración serializada."""
        payload = joblib.load(path)
        self.feature_cols = payload["feature_cols"]
        self.window_size  = payload["window_size"]
        self._is_fitted   = True
        # Re-calcula los índices tras cargar
        self._idx_fast = self.feature_cols.index("sma_fast") if "sma_fast" in self.feature_cols else None
        self._idx_slow = self.feature_cols.index("sma_slow") if "sma_slow" in self.feature_cols else None
        logger.info("BaselineModel cargado desde %s", path)
