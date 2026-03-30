"""
ml_strategy.py
==============
Estrategia de trading genérica que delega la decisión en cualquier BaseModel.

Diseño clave (el "pegamento" entre el mundo ML y el mundo de backtesting):
    - Recibe en __init__: el modelo entrenado Y un dict {timestamp → prediction}.
    - generate_signal() busca el timestamp de la vela actual en el dict.
    - El BacktestEngine llama a generate_signal() exactamente igual que con
      RuleBasedStrategy: no sabe ni le importa que hay un modelo ML por debajo.

Por qué se pre-computan las predicciones (en lugar de predecir en tiempo real):
    - Evita recalcular features en cada vela (costoso).
    - Garantiza que el scaler se aplica una sola vez, correctamente.
    - Facilita el análisis posterior (se puede inspeccionar predictions_map).
    - En paper trading (Hito 4), se reemplazará por predicción incremental.

Mapeo de predicciones a señales:
     1  →  "BUY"
    -1  →  "SELL"
     0  →  "HOLD"
    (sin mapeo conocido)  →  "HOLD"  (comportamiento seguro por defecto)
"""

import logging

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# Mapeo explícito: predicción numérica → señal string
_SIGNAL_MAP: dict[int, str] = {
    1:  "BUY",
    -1: "SELL",
    0:  "HOLD",
}


class MLStrategy(BaseStrategy):
    """
    Estrategia que usa cualquier implementación de BaseModel para generar señales.

    Parameters
    ----------
    model : BaseModel
        Modelo ya entrenado (fit() ya llamado).
    predictions_map : dict[pd.Timestamp, int]
        Diccionario {timestamp → predicción} pre-computado.
        Las claves son los timestamps del DataFrame de backtest.
        Los valores son enteros: 1, -1, 0.
    default_signal : str
        Señal a devolver si el timestamp no está en predictions_map.
        Default: "HOLD" (comportamiento conservador).
    """

    def __init__(
        self,
        model: BaseModel,
        predictions_map: dict,
        default_signal: str = "HOLD",
    ) -> None:
        if not model.is_fitted():
            raise ValueError(
                "El modelo pasado a MLStrategy no ha sido entrenado. "
                "Llama a model.fit() antes de crear MLStrategy."
            )

        self.model           = model
        self.predictions_map = predictions_map
        self.default_signal  = default_signal

        buy_count  = sum(1 for v in predictions_map.values() if v == 1)
        sell_count = sum(1 for v in predictions_map.values() if v == -1)
        hold_count = sum(1 for v in predictions_map.values() if v == 0)

        logger.info(
            "MLStrategy creada | modelo: %s | predicciones: %d "
            "(BUY=%d, SELL=%d, HOLD=%d)",
            model.__class__.__name__,
            len(predictions_map), buy_count, sell_count, hold_count,
        )

    def generate_signal(self, row: pd.Series) -> str:
        """
        Devuelve la señal pre-computada para el timestamp de esta vela.

        Parameters
        ----------
        row : pd.Series
            Fila del DataFrame OHLCV con indicadores. Debe tener 'timestamp'.

        Returns
        -------
        "BUY" | "SELL" | "HOLD"
        """
        timestamp = row.get("timestamp")

        if timestamp is None:
            logger.debug("MLStrategy: fila sin 'timestamp', devolviendo %s", self.default_signal)
            return self.default_signal

        # Normaliza el timestamp para asegurar comparabilidad
        # (a veces hay diferencias de tipo entre Timestamp y np.datetime64)
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        pred = self.predictions_map.get(timestamp)

        if pred is None:
            # Timestamp fuera del período de predicción (ej: período de calentamiento)
            return self.default_signal

        signal = _SIGNAL_MAP.get(int(pred), self.default_signal)
        return signal

    def __repr__(self) -> str:
        return f"MLStrategy(model={self.model.__class__.__name__}, n_predictions={len(self.predictions_map)})"


# ─── Factory function ────────────────────────────────────────────────────────

def build_ml_strategy(
    model: BaseModel,
    X: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> "MLStrategy":
    """
    Función de conveniencia: entrena las predicciones y construye MLStrategy.

    Parameters
    ----------
    model      : BaseModel ya entrenado
    X          : Feature matrix normalizada (test set o full set)
    timestamps : Timestamps correspondientes a cada fila de X

    Returns
    -------
    MLStrategy lista para usar en BacktestEngine.
    """
    predictions = model.predict(X)
    predictions_map = {
        ts: int(pred)
        for ts, pred in zip(timestamps, predictions)
    }
    return MLStrategy(model=model, predictions_map=predictions_map)
