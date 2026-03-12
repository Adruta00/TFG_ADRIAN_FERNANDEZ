"""
portfolio.py
============
Responsabilidad única: gestionar el estado del portfolio durante el backtesting.

Registra:
    - Capital disponible (en USDT)
    - Posición actual (cuántas unidades del activo se tienen)
    - Historial completo de operaciones (lista de dicts)
    - Curva de equity (valor total del portfolio en cada vela)

Decisiones de diseño:
    - Solo gestiona el estado; no decide cuándo comprar/vender (eso es la estrategia).
    - No aplica stop-loss ni position sizing (eso es risk_manager.py en Hito 4).
    - Las operaciones se ejecutan al precio de cierre de la vela de la señal
      (simplificación realista para datos horarios).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Representa una operación completa (entrada + salida)."""
    entry_time:   pd.Timestamp
    exit_time:    Optional[pd.Timestamp]
    entry_price:  float
    exit_price:   Optional[float]
    quantity:     float      # Unidades del activo compradas
    fees_paid:    float      # Comisiones totales (entrada + salida)
    pnl:          float = 0.0   # Profit & Loss realizado al cerrar
    side:         str = "LONG"  # Solo LONG en este hito


class Portfolio:
    """
    Gestiona el capital y las posiciones durante el backtest.

    Parameters
    ----------
    initial_capital : float
        Capital inicial en la moneda de cotización (USDT).
    position_size : float
        Fracción del capital disponible a usar en cada operación (0 < p ≤ 1).
    fees : float
        Comisión por operación expresada como fracción (ej: 0.001 = 0.1%).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        position_size: float = 0.95,
        fees: float = 0.001,
    ) -> None:
        self.initial_capital = initial_capital
        self.position_size   = position_size
        self.fees            = fees

        # Estado actual
        self.cash: float         = initial_capital   # USDT disponibles
        self.holding: float      = 0.0               # Unidades del activo en cartera
        self.entry_price: float  = 0.0               # Precio de la última compra

        # Historial
        self.trades: list[Trade]      = []
        self.equity_curve: list[dict] = []  # [{timestamp, equity}]

    # ------------------------------------------------------------------
    # Acciones de trading
    # ------------------------------------------------------------------

    def buy(self, price: float, timestamp: pd.Timestamp) -> None:
        """
        Ejecuta una compra al precio dado si no hay posición abierta.

        Usa position_size * cash disponible.
        """
        if self.holding > 0:
            logger.debug("BUY ignorado: ya hay posición abierta.")
            return

        amount_to_invest = self.cash * self.position_size
        fee              = amount_to_invest * self.fees
        net_investment   = amount_to_invest - fee
        quantity         = net_investment / price

        self.cash    -= amount_to_invest   # Descuenta el efectivo
        self.holding  = quantity
        self.entry_price = price

        # Registra el trade (exit se completa en sell())
        self.trades.append(
            Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                quantity=quantity,
                fees_paid=fee,
            )
        )

        logger.debug("BUY  @ %.4f | qty=%.6f | cash restante=%.2f",
                     price, quantity, self.cash)

    def sell(self, price: float, timestamp: pd.Timestamp) -> None:
        """
        Cierra la posición actual al precio dado.
        """
        if self.holding <= 0:
            logger.debug("SELL ignorado: no hay posición abierta.")
            return

        gross_value = self.holding * price
        fee         = gross_value * self.fees
        net_value   = gross_value - fee

        self.cash   += net_value
        self.holding = 0.0

        # Completa el último trade abierto
        if self.trades and self.trades[-1].exit_time is None:
            t = self.trades[-1]
            t.exit_time  = timestamp
            t.exit_price = price
            t.fees_paid  += fee
            t.pnl        = net_value - (t.quantity * t.entry_price)

        logger.debug("SELL @ %.4f | valor bruto=%.2f | fee=%.2f | cash=%.2f",
                     price, gross_value, fee, self.cash)

    # ------------------------------------------------------------------
    # Métricas y utilidades
    # ------------------------------------------------------------------

    def record_equity(self, timestamp: pd.Timestamp, price: float) -> None:
        """
        Registra el valor total del portfolio en esta vela.
        Llamar una vez por vela en el bucle del engine.
        """
        equity = self.cash + self.holding * price
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

    def get_equity_series(self) -> pd.Series:
        """Devuelve la curva de equity como pd.Series indexada por timestamp."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        return df["equity"]

    def get_trades_df(self) -> pd.DataFrame:
        """Devuelve el historial de trades como DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "entry_time":  t.entry_time,
                    "exit_time":   t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price":  t.exit_price,
                    "quantity":    t.quantity,
                    "fees_paid":   t.fees_paid,
                    "pnl":         t.pnl,
                    "side":        t.side,
                }
                for t in self.trades
            ]
        )

    @property
    def current_equity(self) -> float:
        """Equity actual (puede ser llamado en cualquier momento)."""
        if self.equity_curve:
            return self.equity_curve[-1]["equity"]
        return self.cash

    def reset(self) -> None:
        """Reinicia el portfolio al estado inicial."""
        self.cash         = self.initial_capital
        self.holding      = 0.0
        self.entry_price  = 0.0
        self.trades       = []
        self.equity_curve = []
