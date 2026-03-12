"""
engine.py
=========
Motor principal de backtesting.

Responsabilidad única: iterar sobre el DataFrame histórico, llamar a la
estrategia en cada vela y actualizar el portfolio con las decisiones.

El engine NO sabe:
    - Qué lógica usa la estrategia (rule-based, ML…)
    - Cómo se calcula el position sizing
    - Qué métricas se calculan al final

Solo sabe: recibo un DataFrame + una BaseStrategy y produzco un Portfolio relleno.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from src.backtesting.portfolio import Portfolio
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Contiene todos los resultados de un backtest."""
    portfolio:    Portfolio
    total_trades: int
    final_equity: float
    pnl_abs:      float    # Ganancia/pérdida absoluta en USDT
    pnl_pct:      float    # Ganancia/pérdida porcentual


class BacktestEngine:
    """
    Motor de backtesting basado en eventos de velas (bar-by-bar).

    Parameters
    ----------
    portfolio : Portfolio
        Instancia de Portfolio con el capital inicial ya configurado.
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio

    def run(self, df: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """
        Ejecuta el backtest iterando sobre cada fila del DataFrame.

        Flujo por vela:
            1. Llama a strategy.generate_signal(row) → "BUY" | "SELL" | "HOLD"
            2. Ejecuta la orden en el portfolio (al precio de cierre)
            3. Registra la equity actual

        Al final, cierra cualquier posición abierta al último precio disponible.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV con indicadores calculados (sin NaN).
            Debe tener columna 'timestamp' de tipo datetime con tz UTC.
        strategy : BaseStrategy
            Cualquier implementación concreta de BaseStrategy.

        Returns
        -------
        BacktestResult
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío. Descarga datos antes de hacer backtest.")

        # Reinicia estado para poder ejecutar el mismo engine múltiples veces
        self.portfolio.reset()

        n_rows     = len(df)
        n_buy      = 0
        n_sell     = 0
        n_hold     = 0

        logger.info(
            "Iniciando backtest | %d velas | estrategia: %s",
            n_rows, strategy,
        )

        for idx, row in df.iterrows():
            signal = strategy.generate_signal(row)

            price     = float(row["close"])
            timestamp = row["timestamp"]

            if signal == "BUY":
                self.portfolio.buy(price, timestamp)
                n_buy += 1

            elif signal == "SELL":
                self.portfolio.sell(price, timestamp)
                n_sell += 1

            else:
                n_hold += 1

            # Registra equity después de ejecutar la orden
            self.portfolio.record_equity(timestamp, price)

            # Log de progreso cada 1000 velas
            if (idx + 1) % 1000 == 0:
                logger.debug("  Procesadas %d / %d velas...", idx + 1, n_rows)

        # Cierra posición abierta al precio de la última vela (liquidación)
        last_row = df.iloc[-1]
        if self.portfolio.holding > 0:
            logger.info(
                "Cerrando posición abierta al final del período @ %.4f",
                last_row["close"],
            )
            self.portfolio.sell(float(last_row["close"]), last_row["timestamp"])

        # Calcula métricas de resumen
        final_equity = self.portfolio.current_equity
        pnl_abs      = final_equity - self.portfolio.initial_capital
        pnl_pct      = (pnl_abs / self.portfolio.initial_capital) * 100
        total_trades = len([t for t in self.portfolio.trades if t.exit_time is not None])

        logger.info(
            "Backtest completado | Trades: %d | BUY: %d | SELL: %d | HOLD: %d",
            total_trades, n_buy, n_sell, n_hold,
        )
        logger.info(
            "Capital inicial: %.2f USDT | Capital final: %.2f USDT | PnL: %+.2f (%.2f%%)",
            self.portfolio.initial_capital, final_equity, pnl_abs, pnl_pct,
        )

        return BacktestResult(
            portfolio=self.portfolio,
            total_trades=total_trades,
            final_equity=final_equity,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
        )
