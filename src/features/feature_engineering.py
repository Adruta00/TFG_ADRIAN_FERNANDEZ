"""
feature_engineering.py
=======================
Responsabilidad única: transformar un DataFrame OHLCV con indicadores calculados
en matrices numéricas (X, y) listas para entrenar modelos de ML.

Pipeline:
    1. Selección de features      — columnas de indicadores técnicos relevantes
    2. Ventana deslizante         — cada muestra = los últimos `window_size` valores
    3. Generación de targets      — etiqueta basada en retorno futuro a N velas
    4. Normalización              — StandardScaler ajustado SOLO sobre train
    5. División temporal          — TimeSeriesSplit (nunca k-fold estándar)

REGLA ANTI-LEAKAGE:
    El scaler SOLO se ajusta (fit) sobre datos de entrenamiento.
    Los datos de validación y test se transforman (transform) sin re-ajustar.
    Esta distinción es la diferencia entre resultados honestos y resultados
    inflados artificialmente.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─── Columnas de indicadores que se usan como features ──────────────────────
# Se excluyen: timestamp, open, high, low, close, volume (precio crudo / identificadores)
# Se incluyen: todos los indicadores calculados por technical_indicators.py
DEFAULT_FEATURE_COLS = [
    "sma_fast", "sma_slow",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "ema",
    "atr",
]


# ─── Dataclass de resultado ──────────────────────────────────────────────────

@dataclass
class FeatureMatrix:
    """
    Resultado de create_feature_matrix().

    Attributes
    ----------
    X           : Matriz de features normalizadas. Shape (n_samples, window*n_features)
    y           : Vector de targets. Shape (n_samples,). Valores: 1=BUY, -1=SELL, 0=HOLD
    timestamps  : Timestamp de la vela "actual" (fin de la ventana) para cada muestra
    feature_cols: Lista de columnas usadas como features
    scaler      : StandardScaler ajustado solo sobre train (para usar en producción)
    X_raw       : Matriz sin normalizar (útil para debugging y el baseline model)
    """
    X:            np.ndarray
    y:            np.ndarray
    timestamps:   pd.DatetimeIndex
    feature_cols: list[str]
    scaler:       StandardScaler
    X_raw:        np.ndarray


# ─── Función principal ───────────────────────────────────────────────────────

def create_feature_matrix(
    df: pd.DataFrame,
    window_size: int = 20,
    target_horizon: int = 5,
    threshold_pct: float = 0.003,
    feature_cols: list[str] | None = None,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None,
) -> FeatureMatrix:
    """
    Transforma el DataFrame en matrices X, y para ML.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV con indicadores calculados (output de apply_all).
        Debe tener columna 'timestamp' y 'close'.
    window_size : int
        Número de velas pasadas que forman cada muestra.
        Ejemplo: window_size=20 → cada muestra = los 20 cierres anteriores de todos los indicadores.
    target_horizon : int
        Número de velas hacia el futuro para calcular el retorno objetivo.
        Ejemplo: target_horizon=5 → "¿sube o baja en 5 velas?"
    threshold_pct : float
        Umbral mínimo de retorno para etiquetar como BUY/SELL.
        Por debajo del umbral → HOLD (0). Evita etiquetar ruido como señal.
        Ejemplo: 0.003 = ±0.3%
    feature_cols : list[str] | None
        Columnas a usar como features. Si None, usa DEFAULT_FEATURE_COLS.
    fit_scaler : bool
        Si True, ajusta un scaler nuevo. Si False, usa el scaler proporcionado.
        Poner False para el conjunto de test/validación.
    scaler : StandardScaler | None
        Scaler ya ajustado (requerido si fit_scaler=False).

    Returns
    -------
    FeatureMatrix con X, y, timestamps, feature_cols, scaler, X_raw.

    Raises
    ------
    ValueError
        Si las columnas de features no están en el DataFrame.
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]

    # Valida que las columnas existen
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas de features no encontradas en el DataFrame: {missing}. "
            f"Asegúrate de llamar a apply_all() antes de create_feature_matrix()."
        )

    # Valida tamaño mínimo del DataFrame
    min_rows = window_size + target_horizon + 1
    if len(df) < min_rows:
        raise ValueError(
            f"El DataFrame tiene {len(df)} filas, pero se necesitan al menos "
            f"{min_rows} (window_size={window_size} + target_horizon={target_horizon} + 1)."
        )

    logger.info(
        "Creando feature matrix | window=%d | horizon=%d | threshold=%.3f%% | features=%d",
        window_size, target_horizon, threshold_pct * 100, len(feature_cols),
    )

    feature_array = df[feature_cols].values   # shape (n_rows, n_features)
    close_prices  = df["close"].values
    # Usamos .iloc[i]["timestamp"] en vez de .values para preservar timezone UTC
    # (df["timestamp"].values devuelve numpy datetime64 sin timezone en pandas 3.x)
    timestamps_series = df["timestamp"]

    X_list:  list[np.ndarray] = []
    y_list:  list[int]        = []
    ts_list: list             = []

    # Rango: desde window_size hasta (n_rows - target_horizon - 1)
    # En t=i, la ventana es [i-window_size, i) y el target mira a i+target_horizon
    for i in range(window_size, len(df) - target_horizon):
        # ── Feature vector: ventana deslizante aplanada ──
        # Forma: [feat1_t-w, feat1_t-w+1, ..., feat1_t-1, feat2_t-w, ..., featN_t-1]
        window_data = feature_array[i - window_size : i]   # shape (window_size, n_features)
        x = window_data.flatten()                          # shape (window_size * n_features,)

        # ── Target: retorno futuro ──
        current_price = close_prices[i]
        future_price  = close_prices[i + target_horizon]
        ret = (future_price - current_price) / current_price

        if ret > threshold_pct:
            label = 1     # BUY
        elif ret < -threshold_pct:
            label = -1    # SELL
        else:
            label = 0     # HOLD

        X_list.append(x)
        y_list.append(label)
        ts_list.append(timestamps_series.iloc[i])  # pd.Timestamp con tz=UTC

    X_raw = np.array(X_list, dtype=np.float64)
    y     = np.array(y_list, dtype=np.int32)
    timestamps = pd.DatetimeIndex(ts_list)

    logger.info(
        "Feature matrix creada | X shape: %s | Clases: BUY=%d SELL=%d HOLD=%d",
        X_raw.shape,
        (y == 1).sum(), (y == -1).sum(), (y == 0).sum(),
    )

    # ── Normalización ──────────────────────────────────────────────────────
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        logger.info("StandardScaler ajustado sobre %d muestras.", len(X_raw))
    else:
        if scaler is None:
            raise ValueError(
                "fit_scaler=False requiere pasar un scaler ya ajustado mediante el parámetro 'scaler'."
            )
        X = scaler.transform(X_raw)
        logger.info("StandardScaler aplicado (sin re-ajuste) sobre %d muestras.", len(X_raw))

    # Verifica que no haya NaN tras la normalización
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(
            "%d valores NaN detectados en X tras normalización. "
            "Verifica que apply_all() elimina todas las filas NaN del DataFrame.",
            nan_count,
        )

    return FeatureMatrix(
        X=X,
        y=y,
        timestamps=timestamps,
        feature_cols=feature_cols,
        scaler=scaler,
        X_raw=X_raw,
    )


# ─── División temporal ────────────────────────────────────────────────────────

def temporal_split(
    fm: FeatureMatrix,
    test_size: float = 0.2,
    n_splits: int = 5,
) -> dict:
    """
    Divide la FeatureMatrix en conjuntos train/test respetando el orden temporal.

    Estrategia:
        - Test set: último `test_size` porcentaje del total (siempre el más reciente)
        - Train set: el resto
        - Dentro del train set, genera `n_splits` pliegues con TimeSeriesSplit
          para búsqueda de hiperparámetros / validación cruzada temporal

    Parameters
    ----------
    fm         : FeatureMatrix ya creada
    test_size  : Fracción del total reservada como test (default 0.20 = 20%)
    n_splits   : Número de pliegues para TimeSeriesSplit sobre train

    Returns
    -------
    dict con keys:
        X_train, y_train, ts_train: Datos de entrenamiento
        X_test,  y_test,  ts_test : Datos de test
        tscv                      : TimeSeriesSplit configurado para cross-validation
        split_idx                 : Índice donde empieza el test (para alinear con df)
    """
    n = len(fm.X)
    split_idx = int(n * (1 - test_size))

    logger.info(
        "División temporal: train=%d muestras (%.0f%%) | test=%d muestras (%.0f%%)",
        split_idx, (1 - test_size) * 100,
        n - split_idx, test_size * 100,
    )

    return {
        "X_train":   fm.X[:split_idx],
        "y_train":   fm.y[:split_idx],
        "ts_train":  fm.timestamps[:split_idx],
        "X_test":    fm.X[split_idx:],
        "y_test":    fm.y[split_idx:],
        "ts_test":   fm.timestamps[split_idx:],
        "tscv":      TimeSeriesSplit(n_splits=n_splits),
        "split_idx": split_idx,
    }


# ─── Utilidad: reshape para LSTM (Hito 3) ────────────────────────────────────

def reshape_for_lstm(
    X: np.ndarray,
    window_size: int,
    n_features: int,
) -> np.ndarray:
    """
    Convierte la matriz 2D de features en tensor 3D para LSTM.

    Parameters
    ----------
    X           : np.ndarray de shape (n_samples, window_size * n_features)
    window_size : int
    n_features  : int  (len(feature_cols))

    Returns
    -------
    np.ndarray de shape (n_samples, window_size, n_features)

    Notas
    -----
    Esta función no se usa en el Hito 2 pero se define aquí para no
    tener que modificar feature_engineering.py en el Hito 3.
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, window_size, n_features)
