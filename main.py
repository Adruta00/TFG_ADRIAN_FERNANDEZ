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

# ── Imports del Hito 2 (ML pipeline) ──────────────────────────────────────────
from src.features.feature_engineering import create_feature_matrix, temporal_split
from src.models.baseline_model import BaselineModel
from src.models.random_forest_model import RandomForestModel
from src.strategies.ml_strategy import build_ml_strategy


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

def run_backtest(cfg: dict, force_download: bool = False, model_name: str = "rule_based") -> None:
    """Ejecuta el pipeline completo de backtesting."""

    ex  = cfg["exchange"]
    dt  = cfg["data"]
    bt  = cfg["backtesting"]
    ml  = cfg.get("ml", {})

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
    repo.save_processed(df, f"{symbol.replace('/', '_')}_{timeframe}_processed")

    # ── 4. Rama según el modelo elegido ───────────────────────────────
    if model_name == "rule_based":
        _run_rule_based(df, bt, ex, cfg, ml, repo)

    elif model_name in ("random_forest", "baseline"):
        _run_ml_model(df, bt, ex, cfg, ml, repo, model_name)

    else:
        logging.error("Modelo desconocido: '%s'. Usa: rule_based | random_forest", model_name)
        sys.exit(1)


# =============================================================================
# Sub-flujos por modelo
# =============================================================================

def _run_rule_based(df, bt, ex, cfg, ml, repo) -> None:
    """Flujo original del Hito 1: estrategia SMA crossover."""
    strategy  = RuleBasedStrategy()
    portfolio = Portfolio(
        initial_capital=bt["initial_capital"],
        position_size=bt.get("position_size", 0.95),
        fees=ex["fees"],
    )
    engine = BacktestEngine(portfolio=portfolio)
    engine.run(df=df, strategy=strategy)
    _report_and_save(portfolio, bt["initial_capital"], ex["timeframe"], cfg, tag="rule_based")


def _run_ml_model(df, bt, ex, cfg, ml, repo, model_name: str) -> None:
    """
    Flujo del Hito 2: feature engineering → entrenamiento → backtest ML.

    Pasos:
        1. Crear feature matrix CRUDA (sin normalizar) sobre el DataFrame completo
        2. Dividir en train / test respetando el orden temporal
        3. Ajustar el StandardScaler SOLO sobre train (anti-leakage)
        4. Aplicar la transformación al test set SIN re-ajustar
        5. Entrenar el modelo sobre el train set correctamente escalado
        6. Generar predicciones sobre el test set
        7. Ejecutar backtest sobre las velas del test set con MLStrategy
        8. Reportar métricas y comparar con el benchmark

    ─── POR QUÉ ES CRÍTICO AJUSTAR EL SCALER SOLO EN TRAIN ───────────────────
    Si el scaler se ajusta sobre todos los datos (train + test), la media y
    varianza que calcula incluyen el periodo de test. Esto provoca que los
    features del test set —que en un bull run deberían tener z-scores positivos
    ("mercado por encima de lo normal")— se normalicen hacia cero y parezcan
    "neutrales". El Random Forest los clasifica entonces como HOLD/SELL y
    nunca como BUY. Este efecto explica que el modelo predijera BUY=0 con
    datos de Oct-Dic 2023 (subida del 60% de BTC).
    ───────────────────────────────────────────────────────────────────────────
    """
    from sklearn.preprocessing import StandardScaler

    window_size    = ml.get("window_size", 20)
    target_horizon = ml.get("target_horizon", 5)
    threshold_pct  = ml.get("threshold_pct", 0.003)
    test_size      = ml.get("test_size", 0.20)
    n_splits       = ml.get("n_splits", 5)

    logging.info("─" * 55)
    logging.info("  HITO 2 — Pipeline ML  |  modelo: %s", model_name)
    logging.info("─" * 55)

    # ── 4.1 Feature matrix CRUDA (fit_scaler=True solo para obtener X_raw) ──
    # El scaler interno de fm está ajustado sobre TODOS los datos; lo vamos a
    # desechar y reemplazar por uno ajustado correctamente solo sobre train.
    logging.info("Creando feature matrix (window=%d, horizon=%d)...", window_size, target_horizon)
    fm = create_feature_matrix(
        df=df,
        window_size=window_size,
        target_horizon=target_horizon,
        threshold_pct=threshold_pct,
        fit_scaler=True,   # Crea X_raw; reemplazaremos X a continuación
    )

    # ── 4.2 División temporal train / test (sobre índices de X_raw) ──────────
    split = temporal_split(fm, test_size=test_size, n_splits=n_splits)
    split_idx = split["split_idx"]

    # ── 4.3 CORRECCIÓN ANTI-LEAKAGE: scaler ajustado solo en train ───────────
    proper_scaler = StandardScaler()
    X_train = proper_scaler.fit_transform(fm.X_raw[:split_idx])   # fit + transform sobre train
    X_test  = proper_scaler.transform(fm.X_raw[split_idx:])       # solo transform sobre test

    y_train = split["y_train"]
    y_test  = split["y_test"]
    ts_test = split["ts_test"]

    logging.info(
        "Scaler ajustado SOLO en train (%d muestras). "
        "Clases en train: BUY=%d  SELL=%d  HOLD=%d",
        len(X_train),
        (y_train == 1).sum(), (y_train == -1).sum(), (y_train == 0).sum(),
    )

    # ── 4.4 Instanciar y entrenar el modelo ───────────────────────────────────
    if model_name == "random_forest":
        model = RandomForestModel(
            n_estimators=ml.get("rf_n_estimators", 100),
            max_depth=ml.get("rf_max_depth", 10),
            min_samples_leaf=ml.get("rf_min_samples_leaf", 5),
            random_state=ml.get("rf_random_state", 42),
        )
    else:  # baseline
        model = BaselineModel(
            feature_cols=fm.feature_cols,
            window_size=window_size,
        )

    model.fit(X_train, y_train)

    # Guarda el modelo entrenado
    models_path = Path(ml.get("models_path", "results/models"))
    models_path.mkdir(parents=True, exist_ok=True)
    model_file  = models_path / f"{model_name}_model.joblib"
    model.save(str(model_file))

    # Log importancia de features (solo Random Forest)
    if model_name == "random_forest" and hasattr(model, "get_feature_importance_report"):
        top_features = model.get_feature_importance_report(fm.feature_cols, window_size, top_n=10)
        logging.info("Top 10 features más importantes:")
        for fname, importance in top_features:
            logging.info("  %-35s  %.4f", fname, importance)

    # ── 4.5 Predicciones sobre el test set (con features correctamente escaladas) ─
    ml_strategy = build_ml_strategy(
        model=model,
        X=X_test,          # Usa X_test escalado con scaler de train
        timestamps=ts_test,
    )

    # ── 4.6 Recortar el DataFrame al período de test ──────────────────────────
    test_start_ts = ts_test[0]
    df_test = df[df["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    logging.info(
        "Backtest sobre test set: %d velas (%s → %s)",
        len(df_test),
        df_test["timestamp"].iloc[0].strftime("%Y-%m-%d"),
        df_test["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
    )

    # ── 4.7 Backtest ML ───────────────────────────────────────────────────────
    portfolio_ml = Portfolio(
        initial_capital=bt["initial_capital"],
        position_size=bt.get("position_size", 0.95),
        fees=ex["fees"],
    )
    engine_ml = BacktestEngine(portfolio=portfolio_ml)
    engine_ml.run(df=df_test, strategy=ml_strategy)

    # ── 4.8 Backtest baseline (comparación justa: mismo período) ─────────────
    logging.info("Ejecutando benchmark rule_based sobre el mismo período de test...")
    strategy_rb  = RuleBasedStrategy()
    portfolio_rb = Portfolio(
        initial_capital=bt["initial_capital"],
        position_size=bt.get("position_size", 0.95),
        fees=ex["fees"],
    )
    engine_rb = BacktestEngine(portfolio=portfolio_rb)
    engine_rb.run(df=df_test, strategy=strategy_rb)

    # ── 4.9 Reporte comparativo ───────────────────────────────────────────────
    _print_comparison(
        portfolio_ml=portfolio_ml,
        portfolio_rb=portfolio_rb,
        initial_capital=bt["initial_capital"],
        timeframe=ex["timeframe"],
        model_name=model_name,
    )

    # Guarda resultados del modelo ML
    _report_and_save(portfolio_ml, bt["initial_capital"], ex["timeframe"], cfg, tag=model_name)


def _print_comparison(portfolio_ml, portfolio_rb, initial_capital, timeframe, model_name) -> None:
    """Imprime tabla comparativa ML vs Rule-Based."""
    from src.evaluation.metrics import compute_all, print_report

    equity_ml = portfolio_ml.get_equity_series()
    trades_ml = portfolio_ml.get_trades_df()
    metrics_ml = compute_all(equity_ml, trades_ml, initial_capital, timeframe)

    equity_rb = portfolio_rb.get_equity_series()
    trades_rb = portfolio_rb.get_trades_df()
    metrics_rb = compute_all(equity_rb, trades_rb, initial_capital, timeframe)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  📊  COMPARATIVA: {model_name.upper()} vs RULE-BASED")
    print(sep)
    fmt = "  {:<22} {:>14} {:>14}"
    print(fmt.format("Métrica", model_name.upper(), "RULE_BASED"))
    print("─" * 60)
    print(fmt.format("Capital final (USDT)",
                     f"{metrics_ml['final_equity']:,.2f}",
                     f"{metrics_rb['final_equity']:,.2f}"))
    print(fmt.format("PnL %",
                     f"{metrics_ml['pnl_percentage']:+.2f}%",
                     f"{metrics_rb['pnl_percentage']:+.2f}%"))
    print(fmt.format("Sharpe Ratio",
                     f"{metrics_ml['sharpe_ratio']:.4f}",
                     f"{metrics_rb['sharpe_ratio']:.4f}"))
    print(fmt.format("Max Drawdown",
                     f"{metrics_ml['max_drawdown_pct']:+.2%}",
                     f"{metrics_rb['max_drawdown_pct']:+.2%}"))
    print(fmt.format("Win Rate",
                     f"{metrics_ml['win_rate']:.2%}",
                     f"{metrics_rb['win_rate']:.2%}"))
    print(fmt.format("Total Trades",
                     f"{metrics_ml['total_trades']}",
                     f"{metrics_rb['total_trades']}"))
    print(f"{sep}\n")


def _report_and_save(portfolio, initial_capital, timeframe, cfg, tag="backtest") -> None:
    """Calcula métricas, imprime reporte y guarda CSV."""
    from src.evaluation.metrics import compute_all, print_report

    equity_curve = portfolio.get_equity_series()
    trades_df    = portfolio.get_trades_df()
    metrics      = compute_all(equity_curve, trades_df, initial_capital, timeframe)

    print_report(metrics)

    out = Path(cfg["results"]["output_path"])
    out.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_csv(out / f"trades_{tag}.csv", index=False)
        logging.info("Trades guardados → results/trades_%s.csv", tag)

    equity_curve.reset_index().to_csv(out / f"equity_{tag}.csv", index=False)
    logging.info("Equity guardada  → results/equity_%s.csv", tag)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de Trading Algorítmico con ML — TFG UPM"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper"],
        default="backtest",
        help="Modo de ejecución (default: backtest)",
    )
    parser.add_argument(
        "--model",
        choices=["rule_based", "random_forest", "baseline"],
        default="rule_based",
        help=(
            "Modelo a usar en el backtest (default: rule_based). "
            "  rule_based    — estrategia SMA crossover del Hito 1 "
            "  random_forest — RandomForestClassifier (Hito 2) "
            "  baseline      — rule-based envuelto en interfaz ML (Hito 2, comparación)"
        ),
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
    args = parse_args()
    cfg  = load_config(args.config)

    setup_logging(cfg.get("results", {}).get("log_level", "INFO"))

    logging.info("=" * 60)
    logging.info("  TFG · Trading Algorítmico con ML")
    logging.info("  Modo: %s | Modelo: %s | Config: %s",
                 args.mode.upper(), args.model, args.config)
    logging.info("=" * 60)

    if args.mode == "backtest":
        run_backtest(cfg, force_download=args.force_download, model_name=args.model)

    elif args.mode == "paper":
        logging.error("El modo 'paper' se implementa en el Hito 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
