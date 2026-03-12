"""
technical_indicators.py
=======================
Responsabilidad única: calcular indicadores técnicos sobre un DataFrame OHLCV.

Diseño deliberado:
  - Cada indicador es una función independiente que recibe un DataFrame y
    devuelve ese mismo DataFrame enriquecido con nuevas columnas.
  - Las funciones son puras: no modifican el original (trabajan sobre una copia).
  - apply_all() aplica todos los indicadores de una vez.

Esto facilita añadir, quitar o modificar indicadores sin tocar el resto del código.
"""

import logging
import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


# =============================================================================
# Indicadores individuales
# =============================================================================

def add_sma(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """
    Añade dos medias móviles simples (SMA).

    Columnas añadidas:
        sma_fast  — Media rápida (señal de entrada/salida)
        sma_slow  — Media lenta  (tendencia de fondo)

    La estrategia rule-based usa el cruce de estas dos medias para generar señales.
    """
    df = df.copy()
    df[f"sma_fast"] = ta.sma(df["close"], length=fast)
    df[f"sma_slow"] = ta.sma(df["close"], length=slow)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Añade el Relative Strength Index (RSI).

    Columnas añadidas:
        rsi — Oscilador entre 0 y 100.
              > 70: zona de sobrecompra (posible venta)
              < 30: zona de sobreventa  (posible compra)
    """
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=period)
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Añade el MACD (Moving Average Convergence Divergence).

    Columnas añadidas:
        macd        — Línea MACD (EMA_fast - EMA_slow)
        macd_signal — Línea de señal (EMA del MACD)
        macd_hist   — Histograma (MACD - Signal), útil para medir impulso
    """
    df = df.copy()
    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

    # pandas_ta devuelve columnas con nombres como MACD_12_26_9, etc.
    if macd_df is not None and not macd_df.empty:
        df["macd"]        = macd_df.iloc[:, 0]
        df["macd_hist"]   = macd_df.iloc[:, 1]
        df["macd_signal"] = macd_df.iloc[:, 2]
    else:
        logger.warning("MACD no pudo calcularse (¿datos insuficientes?).")
        df["macd"] = df["macd_hist"] = df["macd_signal"] = float("nan")

    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Añade las Bandas de Bollinger.

    Columnas añadidas:
        bb_upper — Banda superior (SMA + std * desviación)
        bb_mid   — Banda media   (SMA simple)
        bb_lower — Banda inferior (SMA - std * desviación)
        bb_width — Ancho de banda (indicador de volatilidad)
    """
    df = df.copy()
    bb = ta.bbands(df["close"], length=period, std=std)

    if bb is not None and not bb.empty:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_mid"]   = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]
        df["bb_width"] = bb.iloc[:, 3]  # Bandwidth
    else:
        logger.warning("Bollinger Bands no pudo calcularse.")
        df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = df["bb_width"] = float("nan")

    return df


def add_ema(df: pd.DataFrame, period: int = 21) -> pd.DataFrame:
    """
    Añade la Media Móvil Exponencial (EMA).

    Columnas añadidas:
        ema — EMA del precio de cierre. Da más peso a los precios recientes
              que la SMA, respondiendo más rápido a cambios de tendencia.
    """
    df = df.copy()
    df["ema"] = ta.ema(df["close"], length=period)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Añade el Average True Range (ATR).

    Columnas añadidas:
        atr — Medida de volatilidad. Cuánto se mueve el precio en promedio
              en el período. Útil para calcular stop-loss dinámicos.
    """
    df = df.copy()
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


# =============================================================================
# Función de conveniencia: aplica todos los indicadores de una vez
# =============================================================================

def apply_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Aplica todos los indicadores técnicos usando los parámetros del config.yaml.

    Parameters
    ----------
    df  : DataFrame OHLCV limpio (timestamp, open, high, low, close, volume)
    cfg : Diccionario con la sección 'indicators' del config.yaml

    Returns
    -------
    pd.DataFrame con todas las columnas de indicadores añadidas.
    Las primeras filas tendrán NaN (período de calentamiento de cada indicador).
    """
    ind = cfg.get("indicators", {})

    df = add_sma(df, fast=ind.get("sma_fast", 20), slow=ind.get("sma_slow", 50))
    df = add_rsi(df, period=ind.get("rsi_period", 14))
    df = add_macd(
        df,
        fast=ind.get("macd_fast", 12),
        slow=ind.get("macd_slow", 26),
        signal=ind.get("macd_signal", 9),
    )
    df = add_bollinger(
        df,
        period=ind.get("bollinger_period", 20),
        std=ind.get("bollinger_std", 2.0),
    )
    df = add_ema(df, period=ind.get("ema_period", 21))
    df = add_atr(df, period=ind.get("atr_period", 14))

    # Elimina las filas iniciales con NaN (período de calentamiento)
    df = df.dropna().reset_index(drop=True)

    logger.info(
        "Indicadores calculados. Filas con datos completos: %d", len(df)
    )
    return df
