"""
rule_based_strategy.py
======================
Estrategia de trading basada en el cruce de medias móviles simples (SMA crossover).

Lógica de señales:
    BUY  — La SMA rápida cruza por ENCIMA de la SMA lenta
           (momento de entrada: tendencia alcista confirmada)
    SELL — La SMA rápida cruza por DEBAJO de la SMA lenta
           (momento de salida: tendencia bajista confirmada)
    HOLD — No hay cruce: las medias no han cambiado de posición relativa

Por qué esta estrategia:
    - Es el benchmark de referencia más clásico en trading algorítmico.
    - Simple, reproducible y sin parámetros ocultos.
    - Permite comparar directamente contra los modelos ML (Hitos 2 y 3).

Prerequisito:
    El DataFrame recibido debe tener columnas 'sma_fast' y 'sma_slow'
    (calculadas previamente por technical_indicators.add_sma).
"""

import logging

import pandas as pd

from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class RuleBasedStrategy(BaseStrategy):
    """
    Estrategia SMA Crossover.

    Internamente mantiene el estado de la vela anterior para detectar
    el momento exacto del cruce (no solo la posición relativa actual).

    Parameters
    ----------
    Ninguno en este hito. Los períodos de SMA se configuran en config.yaml
    y se calculan antes de pasarlos al backtesting engine.
    """

    def __init__(self) -> None:
        # Almacena la posición relativa de las SMAs en la vela anterior
        # None = primera vela (sin estado previo aún)
        self._prev_fast_above_slow: bool | None = None

    def generate_signal(self, row: pd.Series) -> str:
        """
        Genera señal de trading basada en el cruce de SMA.

        El cruce se detecta comparando la posición relativa actual
        con la posición de la vela anterior.

        Returns "BUY" | "SELL" | "HOLD"
        """
        # Comprueba que las columnas necesarias existen en esta fila
        if "sma_fast" not in row.index or "sma_slow" not in row.index:
            logger.error(
                "La fila no contiene columnas 'sma_fast' / 'sma_slow'. "
                "Asegúrate de llamar a technical_indicators.add_sma() primero."
            )
            return "HOLD"

        # Valores actuales (NaN en el período de calentamiento)
        fast = row["sma_fast"]
        slow = row["sma_slow"]

        if pd.isna(fast) or pd.isna(slow):
            return "HOLD"

        current_fast_above_slow: bool = fast > slow

        # Primera vela: inicializa el estado sin generar señal
        if self._prev_fast_above_slow is None:
            self._prev_fast_above_slow = current_fast_above_slow
            return "HOLD"

        signal = "HOLD"

        # Cruce alcista: la rápida acaba de superar a la lenta → BUY
        if current_fast_above_slow and not self._prev_fast_above_slow:
            signal = "BUY"
            logger.debug("BUY  @ %.2f | sma_fast=%.2f sma_slow=%.2f",
                         row["close"], fast, slow)

        # Cruce bajista: la rápida acaba de caer por debajo de la lenta → SELL
        elif not current_fast_above_slow and self._prev_fast_above_slow:
            signal = "SELL"
            logger.debug("SELL @ %.2f | sma_fast=%.2f sma_slow=%.2f",
                         row["close"], fast, slow)

        # Actualiza estado para la siguiente vela
        self._prev_fast_above_slow = current_fast_above_slow

        return signal

    def reset(self) -> None:
        """Reinicia el estado interno (útil para ejecutar múltiples backtests)."""
        self._prev_fast_above_slow = None
