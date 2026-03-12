"""
test_features.py
================
Tests unitarios para la capa de features (technical_indicators.py).

Principio clave testado: cada función de indicador es idempotente
(no modifica el DataFrame original) y añade exactamente las columnas esperadas.
"""

import pandas as pd
import pytest

from src.features.technical_indicators import (
    add_atr,
    add_bollinger,
    add_ema,
    add_macd,
    add_rsi,
    add_sma,
    apply_all,
)


class TestTechnicalIndicators:
    """Tests para cada indicador individual y para apply_all."""

    # ------------------------------------------------------------------
    # Tests de inmutabilidad (no modifica el original)
    # ------------------------------------------------------------------

    def test_add_sma_does_not_mutate_input(self, sample_ohlcv):
        """add_sma no debe modificar el DataFrame original."""
        original_cols = set(sample_ohlcv.columns)
        _ = add_sma(sample_ohlcv, fast=5, slow=10)
        assert set(sample_ohlcv.columns) == original_cols

    # ------------------------------------------------------------------
    # Tests de columnas añadidas
    # ------------------------------------------------------------------

    def test_add_sma_creates_columns(self, sample_ohlcv):
        df = add_sma(sample_ohlcv, fast=5, slow=10)
        assert "sma_fast" in df.columns
        assert "sma_slow" in df.columns

    def test_add_rsi_creates_column(self, sample_ohlcv):
        df = add_rsi(sample_ohlcv, period=7)
        assert "rsi" in df.columns

    def test_add_macd_creates_columns(self, sample_ohlcv):
        df = add_macd(sample_ohlcv, fast=6, slow=12, signal=5)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_add_bollinger_creates_columns(self, sample_ohlcv):
        df = add_bollinger(sample_ohlcv, period=10, std=2.0)
        assert "bb_upper" in df.columns
        assert "bb_mid" in df.columns
        assert "bb_lower" in df.columns

    def test_add_ema_creates_column(self, sample_ohlcv):
        df = add_ema(sample_ohlcv, period=10)
        assert "ema" in df.columns

    def test_add_atr_creates_column(self, sample_ohlcv):
        df = add_atr(sample_ohlcv, period=7)
        assert "atr" in df.columns

    # ------------------------------------------------------------------
    # Tests de valores lógicos
    # ------------------------------------------------------------------

    def test_rsi_bounded_between_0_and_100(self, sample_ohlcv):
        """RSI siempre debe estar entre 0 y 100 en los valores calculados."""
        df = add_rsi(sample_ohlcv, period=7)
        valid = df["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), \
            f"RSI fuera de rango: min={valid.min():.2f}, max={valid.max():.2f}"

    def test_bollinger_upper_above_lower(self, sample_ohlcv):
        """La banda superior siempre debe estar por encima de la inferior."""
        df = add_bollinger(sample_ohlcv, period=10, std=2.0).dropna()
        assert (df["bb_upper"] >= df["bb_lower"]).all()

    def test_bollinger_mid_between_bands(self, sample_ohlcv):
        """La banda media debe estar entre la superior y la inferior."""
        df = add_bollinger(sample_ohlcv, period=10, std=2.0).dropna()
        assert (df["bb_mid"] <= df["bb_upper"]).all()
        assert (df["bb_mid"] >= df["bb_lower"]).all()

    def test_sma_slow_smoother_than_fast(self, sample_ohlcv):
        """La SMA lenta debe tener menor desviación estándar que la rápida."""
        df = add_sma(sample_ohlcv, fast=5, slow=20).dropna()
        assert df["sma_slow"].std() <= df["sma_fast"].std()

    def test_atr_is_non_negative(self, sample_ohlcv):
        """ATR es una medida de volatilidad: nunca puede ser negativo."""
        df = add_atr(sample_ohlcv, period=7)
        valid = df["atr"].dropna()
        assert (valid >= 0).all()

    # ------------------------------------------------------------------
    # Tests de apply_all
    # ------------------------------------------------------------------

    def test_apply_all_creates_all_indicator_columns(self, sample_ohlcv, sample_config):
        """apply_all debe crear todas las columnas de indicadores."""
        df = apply_all(sample_ohlcv, sample_config)
        expected = {"sma_fast", "sma_slow", "rsi", "macd", "macd_signal",
                    "macd_hist", "bb_upper", "bb_mid", "bb_lower", "ema", "atr"}
        for col in expected:
            assert col in df.columns, f"Columna '{col}' falta en el DataFrame"

    def test_apply_all_removes_nan_rows(self, sample_ohlcv, sample_config):
        """apply_all debe eliminar las filas NaN del período de calentamiento."""
        df = apply_all(sample_ohlcv, sample_config)
        assert df.isnull().sum().sum() == 0, \
            "apply_all no debería dejar filas con NaN"

    def test_apply_all_preserves_original_ohlcv_columns(self, sample_ohlcv, sample_config):
        """Las columnas OHLCV originales deben estar presentes en el resultado."""
        df = apply_all(sample_ohlcv, sample_config)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_apply_all_result_shorter_than_input(self, sample_ohlcv, sample_config):
        """El resultado debe tener menos filas que el input (período calentamiento)."""
        df = apply_all(sample_ohlcv, sample_config)
        assert len(df) < len(sample_ohlcv)
