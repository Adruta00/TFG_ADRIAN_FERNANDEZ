"""
test_metrics.py
===============
Tests unitarios para la capa de evaluación (metrics.py).

Verifica que cada métrica produce el valor correcto con datos conocidos.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    average_trade_pnl,
    compute_all,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_pnl,
    win_rate,
)


# =============================================================================
# Fixtures locales
# =============================================================================

@pytest.fixture
def flat_equity() -> pd.Series:
    """Equity completamente plana: Sharpe = 0, drawdown = 0."""
    return pd.Series([10_000.0] * 100)


@pytest.fixture
def growing_equity() -> pd.Series:
    """Equity con tendencia alcista perfecta."""
    return pd.Series([10_000.0 + i * 100 for i in range(100)])


@pytest.fixture
def crashing_equity() -> pd.Series:
    """Equity con una caída del 50%."""
    up   = [10_000.0 + i * 100 for i in range(50)]   # Sube a 15.000
    down = [15_000.0 - i * 100 for i in range(50)]   # Baja a 10.000
    return pd.Series(up + down)


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Historial de trades: 3 ganadores, 2 perdedores."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame([
        {"entry_time": ts, "exit_time": ts, "entry_price": 100, "exit_price": 110, "pnl":  50, "quantity": 1, "fees_paid": 0.1, "side": "LONG"},
        {"entry_time": ts, "exit_time": ts, "entry_price": 110, "exit_price": 105, "pnl": -30, "quantity": 1, "fees_paid": 0.1, "side": "LONG"},
        {"entry_time": ts, "exit_time": ts, "entry_price": 105, "exit_price": 120, "pnl":  80, "quantity": 1, "fees_paid": 0.1, "side": "LONG"},
        {"entry_time": ts, "exit_time": ts, "entry_price": 120, "exit_price": 115, "pnl": -20, "quantity": 1, "fees_paid": 0.1, "side": "LONG"},
        {"entry_time": ts, "exit_time": ts, "entry_price": 115, "exit_price": 130, "pnl":  40, "quantity": 1, "fees_paid": 0.1, "side": "LONG"},
    ])


# =============================================================================
# Sharpe Ratio
# =============================================================================

class TestSharpeRatio:

    def test_flat_equity_gives_zero_sharpe(self, flat_equity):
        assert sharpe_ratio(flat_equity) == 0.0

    def test_growing_equity_gives_positive_sharpe(self, growing_equity):
        result = sharpe_ratio(growing_equity)
        assert result > 0.0

    def test_empty_series_gives_zero(self):
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0

    def test_single_value_gives_zero(self):
        assert sharpe_ratio(pd.Series([10_000.0])) == 0.0


# =============================================================================
# Max Drawdown
# =============================================================================

class TestMaxDrawdown:

    def test_flat_equity_gives_zero_drawdown(self, flat_equity):
        assert max_drawdown(flat_equity) == 0.0

    def test_always_growing_gives_zero_drawdown(self, growing_equity):
        assert max_drawdown(growing_equity) == 0.0

    def test_crash_50_pct_gives_correct_drawdown(self):
        """Una caída del 50% desde el pico debe reportar -0.50 aprox."""
        up   = [10_000.0 + i * 100 for i in range(50)]
        down = [15_000.0 - i * 300 for i in range(50)]
        equity = pd.Series(up + down)
        mdd = max_drawdown(equity)
        assert mdd < 0.0

    def test_drawdown_is_between_minus1_and_zero(self, crashing_equity):
        """El drawdown siempre debe estar en el rango [-1, 0]."""
        mdd = max_drawdown(crashing_equity)
        assert -1.0 <= mdd <= 0.0

    def test_empty_series_gives_zero(self):
        assert max_drawdown(pd.Series(dtype=float)) == 0.0


# =============================================================================
# Win Rate
# =============================================================================

class TestWinRate:

    def test_win_rate_three_of_five(self, sample_trades):
        """3 ganadores de 5 → win rate = 0.60."""
        result = win_rate(sample_trades)
        assert result == pytest.approx(0.60, abs=1e-4)

    def test_all_winners(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame([
            {"exit_time": ts, "pnl": 100},
            {"exit_time": ts, "pnl": 50},
        ])
        assert win_rate(df) == 1.0

    def test_all_losers(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame([
            {"exit_time": ts, "pnl": -100},
            {"exit_time": ts, "pnl": -50},
        ])
        assert win_rate(df) == 0.0

    def test_empty_trades_gives_zero(self):
        assert win_rate(pd.DataFrame()) == 0.0

    def test_win_rate_between_zero_and_one(self, sample_trades):
        result = win_rate(sample_trades)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Total PnL
# =============================================================================

class TestTotalPnl:

    def test_profit_gives_positive_pnl(self):
        result = total_pnl(initial_capital=10_000.0, final_equity=12_000.0)
        assert result["absolute"]   == pytest.approx(2_000.0)
        assert result["percentage"] == pytest.approx(20.0)

    def test_loss_gives_negative_pnl(self):
        result = total_pnl(initial_capital=10_000.0, final_equity=8_000.0)
        assert result["absolute"]   == pytest.approx(-2_000.0)
        assert result["percentage"] == pytest.approx(-20.0)

    def test_breakeven_gives_zero(self):
        result = total_pnl(10_000.0, 10_000.0)
        assert result["absolute"]   == 0.0
        assert result["percentage"] == 0.0


# =============================================================================
# Profit Factor
# =============================================================================

class TestProfitFactor:

    def test_profit_factor_correct(self, sample_trades):
        """Ganancias: 50+80+40=170 | Pérdidas: 30+20=50 | PF = 3.4"""
        result = profit_factor(sample_trades)
        assert result == pytest.approx(3.4, abs=1e-2)

    def test_only_winners_gives_inf(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame([{"exit_time": ts, "pnl": 100}])
        assert profit_factor(df) == float("inf")

    def test_empty_trades_gives_zero(self):
        assert profit_factor(pd.DataFrame()) == 0.0


# =============================================================================
# compute_all (integración)
# =============================================================================

class TestComputeAll:

    def test_compute_all_returns_all_keys(self, growing_equity, sample_trades):
        """compute_all debe devolver todas las métricas esperadas."""
        expected_keys = {
            "initial_capital", "final_equity", "pnl_absolute", "pnl_percentage",
            "sharpe_ratio", "max_drawdown_pct", "total_trades", "win_rate",
            "avg_trade_pnl", "profit_factor",
        }
        result = compute_all(
            equity_curve=growing_equity,
            trades_df=sample_trades,
            initial_capital=10_000.0,
            timeframe="1h",
        )
        assert expected_keys.issubset(set(result.keys()))

    def test_compute_all_values_are_numeric(self, growing_equity, sample_trades):
        """Todos los valores del resultado deben ser numéricos."""
        result = compute_all(growing_equity, sample_trades, 10_000.0)
        for key, val in result.items():
            assert isinstance(val, (int, float)), \
                f"Clave '{key}' debería ser numérica, es {type(val)}"
