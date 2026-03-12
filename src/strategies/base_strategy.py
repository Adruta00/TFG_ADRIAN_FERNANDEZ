"""
base_strategy.py
================
Define el contrato que TODA estrategia del sistema debe cumplir.

Por qué una clase abstracta:
    - BacktestEngine solo necesita saber que cualquier objeto que recibe
      tiene un método generate_signal(). No le importa si es rule-based o ML.
    - Añadir una nueva estrategia = crear un archivo nuevo que herede de
      BaseStrategy. El engine no cambia.

Señales posibles:
    "BUY"  — Abrir posición larga (comprar)
    "SELL" — Cerrar posición / abrir posición corta (vender)
    "HOLD" — No hacer nada en esta vela
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Clase abstracta que define la interfaz de todas las estrategias de trading.

    Cualquier clase concreta DEBE implementar generate_signal().
    """

    @abstractmethod
    def generate_signal(self, row: pd.Series) -> str:
        """
        Genera una señal de trading para una vela (fila del DataFrame).

        Parameters
        ----------
        row : pd.Series
            Una fila del DataFrame OHLCV con indicadores calculados.
            Contiene: timestamp, open, high, low, close, volume,
                      sma_fast, sma_slow, rsi, macd, bb_upper, bb_lower, ema, atr, etc.

        Returns
        -------
        str
            Una de: "BUY" | "SELL" | "HOLD"
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
