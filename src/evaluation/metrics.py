"""
metrics.py
==========
Responsabilidad única: calcular métricas de rendimiento y riesgo.

Todas las funciones son puras: reciben datos y devuelven valores.
No modifican ningún estado externo.

Métricas implementadas en el Hito 1:
    - Sharpe Ratio       — Rentabilidad ajustada al riesgo
    - Maximum Drawdown   — Pérdida máxima desde un pico (riesgo)
    - Win Rate           — Porcentaje de trades cerrados en beneficio
    - Total PnL          — Ganancia/pérdida total absoluta y porcentual

Añadir nuevas métricas en hitos futuros = añadir funciones aquí. El resto no cambia.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Funciones individuales
# =============================================================================

def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Calcula el Sharpe Ratio anualizado (asumiendo tasa libre de riesgo = 0).

    Fórmula: (media de retornos / desviación estándar de retornos) * sqrt(períodos/año)

    Parameters
    ----------
    equity_curve : pd.Series
        Serie con el valor total del portfolio en cada vela (curva de equity).
    periods_per_year : int
        Número de períodos en un año. Para velas de 1h = 8760 (24*365).
        Para 4h = 2190. Para 1d = 365.

    Returns
    -------
    float
        Sharpe Ratio anualizado. > 1.0 se considera aceptable; > 2.0, bueno.
        Devuelve 0.0 si no hay suficientes datos.
    """
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()

    if returns.std() == 0:
        return 0.0

    ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    return float(round(ratio, 4))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calcula el Maximum Drawdown (MDD) como porcentaje.

    El MDD es la mayor caída desde un pico hasta un valle subsecuente.
    Es la métrica de riesgo más importante en trading.

    Returns
    -------
    float
        MDD en porcentaje (negativo). Ej: -0.35 = caída máxima del 35%.
        Mejor cuanto más cercano a 0.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak     = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    mdd      = float(drawdown.min())
    return round(mdd, 4)


def win_rate(trades_df: pd.DataFrame) -> float:
    """
    Calcula el porcentaje de trades cerrados que resultaron en ganancia.

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame con columna 'pnl' (solo trades cerrados, exit_time no nulo).

    Returns
    -------
    float
        Win rate entre 0.0 y 1.0. Ej: 0.55 = 55% de trades ganadores.
    """
    closed = trades_df[trades_df["exit_time"].notna()]

    if closed.empty:
        return 0.0

    winners = (closed["pnl"] > 0).sum()
    return float(round(winners / len(closed), 4))


def total_pnl(
    initial_capital: float,
    final_equity: float,
) -> dict[str, float]:
    """
    Calcula el PnL total absoluto y porcentual.

    Returns
    -------
    dict con claves 'absolute' (USDT) y 'percentage' (%).
    """
    pnl_abs = final_equity - initial_capital
    pnl_pct = (pnl_abs / initial_capital) * 100
    return {
        "absolute":   round(pnl_abs, 4),
        "percentage": round(pnl_pct, 4),
    }


def average_trade_pnl(trades_df: pd.DataFrame) -> float:
    """PnL medio por trade cerrado en USDT."""
    closed = trades_df[trades_df["exit_time"].notna()]
    if closed.empty:
        return 0.0
    return float(round(closed["pnl"].mean(), 4))


def profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Profit Factor = suma de ganancias brutas / suma de pérdidas brutas (absoluta).
    > 1.0 → el sistema gana más de lo que pierde en total.
    """
    closed   = trades_df[trades_df["exit_time"].notna()]
    gains    = closed[closed["pnl"] > 0]["pnl"].sum()
    losses   = abs(closed[closed["pnl"] < 0]["pnl"].sum())

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return float(round(gains / losses, 4))


# =============================================================================
# Función de conveniencia: calcula y muestra todas las métricas
# =============================================================================

def compute_all(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float,
    timeframe: str = "1h",
) -> dict:
    """
    Calcula todas las métricas y devuelve un diccionario de resultados.

    Parameters
    ----------
    equity_curve    : Curva de equity del portfolio
    trades_df       : DataFrame de trades cerrados
    initial_capital : Capital inicial en USDT
    timeframe       : Timeframe de las velas (para anualización del Sharpe)

    Returns
    -------
    dict con todas las métricas calculadas.
    """
    _periods_map = {"1m": 525600, "5m": 105120, "15m": 35040,
                    "1h": 8760, "4h": 2190, "1d": 365}
    periods = _periods_map.get(timeframe, 8760)

    final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital

    results = {
        "initial_capital":  round(initial_capital, 2),
        "final_equity":     round(final_equity, 2),
        "pnl_absolute":     total_pnl(initial_capital, final_equity)["absolute"],
        "pnl_percentage":   total_pnl(initial_capital, final_equity)["percentage"],
        "sharpe_ratio":     sharpe_ratio(equity_curve, periods_per_year=periods),
        "max_drawdown_pct": max_drawdown(equity_curve),
        "total_trades":     len(trades_df[trades_df["exit_time"].notna()]) if not trades_df.empty else 0,
        "win_rate":         win_rate(trades_df) if not trades_df.empty else 0.0,
        "avg_trade_pnl":    average_trade_pnl(trades_df) if not trades_df.empty else 0.0,
        "profit_factor":    profit_factor(trades_df) if not trades_df.empty else 0.0,
    }

    return results


def print_report(metrics: dict) -> None:
    """Imprime un informe formateado de métricas en consola."""
    sep = "─" * 50
    print(f"\n{sep}")
    print("  📊  INFORME DE RESULTADOS — BACKTEST")
    print(sep)
    print(f"  Capital inicial :  {metrics['initial_capital']:>12,.2f} USDT")
    print(f"  Capital final   :  {metrics['final_equity']:>12,.2f} USDT")
    print(f"  PnL absoluto    :  {metrics['pnl_absolute']:>+12,.2f} USDT")
    print(f"  PnL %           :  {metrics['pnl_percentage']:>+11.2f}%")
    print(sep)
    print(f"  Sharpe Ratio    :  {metrics['sharpe_ratio']:>12.4f}")
    print(f"  Max Drawdown    :  {metrics['max_drawdown_pct']:>+11.2%}")
    print(f"  Win Rate        :  {metrics['win_rate']:>11.2%}")
    print(f"  Total Trades    :  {metrics['total_trades']:>12d}")
    print(f"  PnL medio/trade :  {metrics['avg_trade_pnl']:>+12.4f} USDT")
    print(f"  Profit Factor   :  {metrics['profit_factor']:>12.4f}")
    print(f"{sep}\n")
