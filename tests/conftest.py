"""
conftest.py
===========
Fixtures compartidas por todos los tests.

pytest las detecta automáticamente en este archivo.
No es necesario importar conftest.py: pytest lo inyecta solo.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    DataFrame OHLCV sintético con 200 velas.

    Simula un mercado con tendencia alcista seguida de bajista,
    suficiente para que todos los indicadores se calculen (period > 50).
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)  # Reproducibilidad

    n = 200
    timestamps = pd.date_range(
        start="2024-01-01", periods=n, freq="1h", tz="UTC"
    )

    # Precio sintético: tendencia + ruido
    prices = 40_000 + np.cumsum(rng.normal(0, 200, n))
    prices = np.clip(prices, 1, None)  # Sin precios negativos

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open":   prices * (1 + rng.normal(0, 0.001, n)),
            "high":   prices * (1 + abs(rng.normal(0, 0.005, n))),
            "low":    prices * (1 - abs(rng.normal(0, 0.005, n))),
            "close":  prices,
            "volume": rng.uniform(100, 1000, n),
        }
    )
    return df


@pytest.fixture
def sample_config() -> dict:
    """Configuración mínima para tests (espeja config.yaml)."""
    return {
        "exchange": {
            "id": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "fees": 0.001,
        },
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "raw_path": "/tmp/tfg_test/raw",
            "processed_path": "/tmp/tfg_test/processed",
        },
        "backtesting": {
            "initial_capital": 10_000.0,
            "position_size": 0.95,
        },
        "indicators": {
            "sma_fast": 5,     # Períodos cortos para que calcule con 200 velas
            "sma_slow": 10,
            "rsi_period": 7,
            "macd_fast": 6,
            "macd_slow": 12,
            "macd_signal": 5,
            "bollinger_period": 10,
            "bollinger_std": 2.0,
            "ema_period": 10,
            "atr_period": 7,
        },
        "results": {
            "output_path": "/tmp/tfg_test/results",
            "log_level": "WARNING",
        },
    }
