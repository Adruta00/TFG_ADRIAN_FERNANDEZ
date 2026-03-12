"""
test_backtesting.py
===================
Tests unitarios para la capa de backtesting y estrategias.

Cubre:
    - Portfolio: buy/sell, cálculo de equity, historial de trades
    - RuleBasedStrategy: generación correcta de señales BUY/SELL/HOLD
    - BacktestEngine: que el bucle funcione end-to-end con datos sintéticos
"""

import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.backtesting.portfolio import Portfolio
from src.features.technical_indicators import apply_all
from src.strategies.rule_based_strategy import RuleBasedStrategy


# =============================================================================
# Portfolio Tests
# =============================================================================

class TestPortfolio:

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        return Portfolio(initial_capital=10_000.0, position_size=0.95, fees=0.001)

    @pytest.fixture
    def ts(self) -> pd.Timestamp:
        return pd.Timestamp("2024-01-01 10:00:00", tz="UTC")

    # ------------------------------------------------------------------
    def test_initial_state(self, portfolio):
        assert portfolio.cash    == 10_000.0
        assert portfolio.holding == 0.0
        assert len(portfolio.trades) == 0

    def test_buy_reduces_cash(self, portfolio, ts):
        portfolio.buy(price=40_000.0, timestamp=ts)
        assert portfolio.cash < 10_000.0

    def test_buy_increases_holding(self, portfolio, ts):
        portfolio.buy(price=40_000.0, timestamp=ts)
        assert portfolio.holding > 0.0

    def test_sell_after_buy_zeroes_holding(self, portfolio, ts):
        t2 = pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        portfolio.buy(price=40_000.0, timestamp=ts)
        portfolio.sell(price=41_000.0, timestamp=t2)
        assert portfolio.holding == 0.0

    def test_buy_twice_ignores_second_buy(self, portfolio, ts):
        """No debe abrir una segunda posición si ya hay una abierta."""
        t2 = pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        portfolio.buy(price=40_000.0, timestamp=ts)
        holdings_after_first = portfolio.holding
        portfolio.buy(price=41_000.0, timestamp=t2)
        assert portfolio.holding == holdings_after_first  # No cambia

    def test_sell_without_position_does_nothing(self, portfolio, ts):
        initial_cash = portfolio.cash
        portfolio.sell(price=40_000.0, timestamp=ts)
        assert portfolio.cash == initial_cash

    def test_profitable_trade_increases_cash(self, portfolio, ts):
        t2 = pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        portfolio.buy(price=40_000.0, timestamp=ts)
        portfolio.sell(price=50_000.0, timestamp=t2)
        # Después de vender a mayor precio, el cash debe ser > 10000
        assert portfolio.cash > portfolio.initial_capital

    def test_loss_trade_decreases_cash(self, portfolio, ts):
        t2 = pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        portfolio.buy(price=40_000.0, timestamp=ts)
        portfolio.sell(price=30_000.0, timestamp=t2)
        assert portfolio.cash < portfolio.initial_capital

    def test_trade_pnl_is_recorded(self, portfolio, ts):
        t2 = pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        portfolio.buy(price=40_000.0, timestamp=ts)
        portfolio.sell(price=44_000.0, timestamp=t2)
        assert len(portfolio.trades) == 1
        assert portfolio.trades[0].pnl != 0.0

    def test_equity_curve_grows_with_records(self, portfolio, ts):
        for i in range(5):
            t = pd.Timestamp(f"2024-01-01 {10+i:02d}:00:00", tz="UTC")
            portfolio.record_equity(t, price=40_000.0)
        assert len(portfolio.equity_curve) == 5

    def test_reset_restores_initial_state(self, portfolio, ts):
        t2 = pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        portfolio.buy(price=40_000.0, timestamp=ts)
        portfolio.sell(price=41_000.0, timestamp=t2)
        portfolio.reset()

        assert portfolio.cash    == portfolio.initial_capital
        assert portfolio.holding == 0.0
        assert len(portfolio.trades) == 0
        assert len(portfolio.equity_curve) == 0

    def test_fees_are_deducted_on_buy(self, portfolio, ts):
        """La compra debe descontar las comisiones."""
        portfolio.buy(price=40_000.0, timestamp=ts)
        expected_qty = (10_000.0 * 0.95 * (1 - 0.001)) / 40_000.0
        assert abs(portfolio.holding - expected_qty) < 1e-8


# =============================================================================
# Strategy Tests
# =============================================================================

class TestRuleBasedStrategy:

    def _make_row(self, sma_fast: float, sma_slow: float, price: float = 40_000.0) -> pd.Series:
        """Crea una fila mínima para testear la estrategia."""
        return pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "close":    price,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
        })

    def test_first_row_returns_hold(self):
        """La primera vela nunca debe generar señal (sin estado previo)."""
        strategy = RuleBasedStrategy()
        row = self._make_row(sma_fast=100, sma_slow=90)
        assert strategy.generate_signal(row) == "HOLD"

    def test_bullish_crossover_generates_buy(self):
        """Cruce alcista (fast sube sobre slow) → BUY."""
        strategy = RuleBasedStrategy()

        # Vela 1: fast DEBAJO de slow
        strategy.generate_signal(self._make_row(sma_fast=90, sma_slow=100))

        # Vela 2: fast ENCIMA de slow (cruce alcista)
        signal = strategy.generate_signal(self._make_row(sma_fast=110, sma_slow=100))
        assert signal == "BUY"

    def test_bearish_crossover_generates_sell(self):
        """Cruce bajista (fast cae bajo slow) → SELL."""
        strategy = RuleBasedStrategy()

        # Vela 1: fast ENCIMA de slow
        strategy.generate_signal(self._make_row(sma_fast=110, sma_slow=100))

        # Vela 2: fast DEBAJO de slow (cruce bajista)
        signal = strategy.generate_signal(self._make_row(sma_fast=90, sma_slow=100))
        assert signal == "SELL"

    def test_no_crossover_returns_hold(self):
        """Sin cruce → HOLD."""
        strategy = RuleBasedStrategy()

        strategy.generate_signal(self._make_row(sma_fast=110, sma_slow=100))  # fast > slow
        signal = strategy.generate_signal(self._make_row(sma_fast=115, sma_slow=100))  # sigue fast > slow
        assert signal == "HOLD"

    def test_nan_values_return_hold(self):
        """Si las SMAs son NaN (período de calentamiento) → HOLD."""
        import math
        strategy = RuleBasedStrategy()
        row = self._make_row(sma_fast=float("nan"), sma_slow=float("nan"))
        assert strategy.generate_signal(row) == "HOLD"

    def test_missing_columns_return_hold(self):
        """Si faltan las columnas sma_fast/sma_slow → HOLD (sin crash)."""
        strategy = RuleBasedStrategy()
        row = pd.Series({"close": 40_000.0})
        assert strategy.generate_signal(row) == "HOLD"

    def test_only_three_possible_signals(self):
        """generate_signal solo puede devolver BUY, SELL o HOLD."""
        strategy = RuleBasedStrategy()
        rows = [
            self._make_row(90, 100),
            self._make_row(110, 100),
            self._make_row(105, 100),
            self._make_row(80, 100),
            self._make_row(85, 100),
        ]
        for row in rows:
            signal = strategy.generate_signal(row)
            assert signal in {"BUY", "SELL", "HOLD"}, f"Señal inválida: {signal}"

    def test_reset_clears_state(self):
        """Después de reset(), la primera señal vuelve a ser HOLD."""
        strategy = RuleBasedStrategy()
        strategy.generate_signal(self._make_row(110, 100))
        strategy.reset()
        signal = strategy.generate_signal(self._make_row(90, 100))
        assert signal == "HOLD"


# =============================================================================
# BacktestEngine Integration Test
# =============================================================================

class TestBacktestEngine:

    def test_engine_runs_end_to_end(self, sample_ohlcv, sample_config):
        """El engine debe completar un backtest sin errores."""
        df = apply_all(sample_ohlcv, sample_config)
        strategy  = RuleBasedStrategy()
        portfolio = Portfolio(initial_capital=10_000.0, position_size=0.95, fees=0.001)
        engine    = BacktestEngine(portfolio=portfolio)

        result = engine.run(df=df, strategy=strategy)

        assert result is not None
        assert result.final_equity > 0
        assert len(portfolio.equity_curve) == len(df)

    def test_equity_curve_has_same_length_as_data(self, sample_ohlcv, sample_config):
        """La curva de equity debe tener una entrada por cada vela."""
        df = apply_all(sample_ohlcv, sample_config)
        portfolio = Portfolio(initial_capital=10_000.0)
        engine    = BacktestEngine(portfolio=portfolio)

        engine.run(df=df, strategy=RuleBasedStrategy())
        assert len(portfolio.equity_curve) == len(df)

    def test_no_open_position_at_end(self, sample_ohlcv, sample_config):
        """El engine debe cerrar posiciones abiertas al final del período."""
        df = apply_all(sample_ohlcv, sample_config)
        portfolio = Portfolio(initial_capital=10_000.0)
        engine    = BacktestEngine(portfolio=portfolio)

        engine.run(df=df, strategy=RuleBasedStrategy())
        assert portfolio.holding == 0.0

    def test_engine_raises_on_empty_dataframe(self):
        """El engine debe lanzar ValueError con un DataFrame vacío."""
        portfolio = Portfolio(initial_capital=10_000.0)
        engine    = BacktestEngine(portfolio=portfolio)

        with pytest.raises(ValueError, match="vacío"):
            engine.run(df=pd.DataFrame(), strategy=RuleBasedStrategy())

    def test_multiple_runs_are_independent(self, sample_ohlcv, sample_config):
        """Ejecutar el backtest dos veces debe producir resultados idénticos."""
        df        = apply_all(sample_ohlcv, sample_config)
        portfolio = Portfolio(initial_capital=10_000.0)
        engine    = BacktestEngine(portfolio=portfolio)
        strategy  = RuleBasedStrategy()

        result1 = engine.run(df=df, strategy=strategy)
        strategy.reset()
        result2 = engine.run(df=df, strategy=strategy)

        assert abs(result1.final_equity - result2.final_equity) < 1e-6
