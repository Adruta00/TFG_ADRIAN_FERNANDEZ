"""
main.py
=======
Punto de entrada principal del sistema de trading algorítmico.

Orquesta el flujo completo del Hito 1:
    1. Carga la configuración desde config/config.yaml
    2. Descarga (o carga desde caché) los datos OHLCV históricos
    3. Calcula los indicadores técnicos
    4. Crea la estrategia y el portfolio
    5. Ejecuta el backtest
    6. Calcula y muestra las métricas

Uso:
    python main.py --mode backtest
    python main.py --mode backtest --config config/config.yaml
    python main.py --mode backtest --force-download   (ignora caché)

El flag --mode permite preparar la extensión a paper trading en el Hito 4
sin romper este código.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Asegura que el raíz del proyecto esté en el path de Python
sys.path.insert(0, str(Path(__file__).parent))

from src.backtesting.engine import BacktestEngine
from src.backtesting.portfolio import Portfolio
from src.data.data_repository import DataRepository
from src.data.historical_loader import HistoricalLoader
from src.evaluation.metrics import compute_all, print_report
from src.features.technical_indicators import apply_all
from src.strategies.rule_based_strategy import RuleBasedStrategy


# =============================================================================
# Configuración del sistema de logging
# =============================================================================

def setup_logging(level: str = "INFO") -> None:
    """Configura el logger raíz con formato legible."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Carga de configuración
# =============================================================================

def load_config(config_path: str) -> dict:
    """Lee el archivo YAML y devuelve el diccionario de configuración."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =============================================================================
# Flujo principal: modo backtest
# =============================================================================

def run_backtest(cfg: dict, force_download: bool = False) -> None:
    """Ejecuta el pipeline completo de backtesting."""

    ex  = cfg["exchange"]
    dt  = cfg["data"]
    bt  = cfg["backtesting"]

    # ── 1. Repositorio de datos ────────────────────────────────────────
    repo = DataRepository(
        raw_path=dt["raw_path"],
        processed_path=dt["processed_path"],
    )

    # ── 2. Datos OHLCV (caché o descarga) ─────────────────────────────
    symbol    = ex["symbol"]
    timeframe = ex["timeframe"]
    start     = dt["start_date"]
    end       = dt["end_date"]

    if not force_download and repo.ohlcv_exists(symbol, timeframe, start, end):
        logging.info("Cargando datos desde caché local...")
        df_raw = repo.load_ohlcv(symbol, timeframe, start, end)
    else:
        logging.info("Descargando datos desde %s...", ex["id"])
        loader = HistoricalLoader(
            exchange_id=ex["id"],
            symbol=symbol,
            timeframe=timeframe,
        )
        df_raw = loader.fetch(start_date=start, end_date=end)
        repo.save_ohlcv(df_raw, symbol, timeframe, start, end)

    if df_raw is None or df_raw.empty:
        logging.error("No hay datos disponibles. Abortando.")
        sys.exit(1)

    logging.info("Datos cargados: %d velas | %s → %s",
                 len(df_raw),
                 df_raw["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                 df_raw["timestamp"].iloc[-1].strftime("%Y-%m-%d"))

    # ── 3. Indicadores técnicos ────────────────────────────────────────
    logging.info("Calculando indicadores técnicos...")
    df = apply_all(df_raw, cfg)

    # Guarda los datos procesados para uso en notebooks
    repo.save_processed(df, f"{symbol.replace('/', '_')}_{timeframe}_processed")

    # ── 4. Estrategia y portfolio ──────────────────────────────────────
    strategy = RuleBasedStrategy()

    portfolio = Portfolio(
        initial_capital=bt["initial_capital"],
        position_size=bt.get("position_size", 0.95),
        fees=ex["fees"],
    )

    # ── 5. Backtest ────────────────────────────────────────────────────
    engine = BacktestEngine(portfolio=portfolio)
    result = engine.run(df=df, strategy=strategy)

    # ── 6. Métricas y reporte ──────────────────────────────────────────
    equity_curve = portfolio.get_equity_series()
    trades_df    = portfolio.get_trades_df()

    metrics = compute_all(
        equity_curve=equity_curve,
        trades_df=trades_df,
        initial_capital=bt["initial_capital"],
        timeframe=timeframe,
    )

    print_report(metrics)

    # Guarda el historial de trades para análisis posterior
    if not trades_df.empty:
        trades_path = Path(cfg["results"]["output_path"]) / "trades_history.csv"
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_path, index=False)
        logging.info("Historial de trades guardado → %s", trades_path)

    # Guarda la curva de equity
    equity_path = Path(cfg["results"]["output_path"]) / "equity_curve.csv"
    equity_curve.reset_index().to_csv(equity_path, index=False)
    logging.info("Curva de equity guardada → %s", equity_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de Trading Algorítmico con ML — TFG UPM"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper"],   # 'paper' se implementa en Hito 4
        default="backtest",
        help="Modo de ejecución (default: backtest)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Fuerza la re-descarga de datos aunque existan en caché",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)

    setup_logging(cfg.get("results", {}).get("log_level", "INFO"))

    logging.info("=" * 60)
    logging.info("  TFG · Trading Algorítmico con ML · Hito 1")
    logging.info("  Modo: %s | Config: %s", args.mode.upper(), args.config)
    logging.info("=" * 60)

    if args.mode == "backtest":
        run_backtest(cfg, force_download=args.force_download)

    elif args.mode == "paper":
        # Reservado para el Hito 4 (paper_trader.py)
        logging.error("El modo 'paper' se implementa en el Hito 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
