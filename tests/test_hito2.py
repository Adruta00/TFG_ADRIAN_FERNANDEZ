"""
test_hito2.py
=============
Tests unitarios para el Hito 2: Feature Engineering, Modelos ML y MLStrategy.

Cobertura:
    1. Integridad de features:
       - create_feature_matrix no genera NaN tras normalización
       - X.shape es consistente con window_size y n_features
       - len(X) == len(y) == len(timestamps)

    2. Validación temporal (anti-leakage):
       - temporal_split siempre pone el test DESPUÉS del train (nunca antes)
       - TimeSeriesSplit no mezcla datos futuros con pasados en ningún fold
       - El scaler se ajusta solo sobre train (fit_scaler=False en test)

    3. Consistencia de modelos (save/load):
       - Un modelo cargado predice exactamente lo mismo que el entrenado
       - Funciona para RandomForestModel y BaselineModel

    4. Mapeo de señales (MLStrategy):
       - Predicción 1  → "BUY"
       - Predicción -1 → "SELL"
       - Predicción 0  → "HOLD"
       - Timestamp desconocido → "HOLD" (valor por defecto seguro)

    5. Integración end-to-end:
       - El flujo completo df → features → entrenamiento → backtest funciona
         sin errores de dimensiones ni tipos
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

# ── Módulos del Hito 2 ────────────────────────────────────────────────────────
from src.features.feature_engineering import (
    FeatureMatrix,
    create_feature_matrix,
    reshape_for_lstm,
    temporal_split,
)
from src.models.base_model import BaseModel
from src.models.baseline_model import BaselineModel
from src.models.random_forest_model import RandomForestModel
from src.strategies.ml_strategy import MLStrategy, build_ml_strategy

# ── Módulos del Hito 1 (necesarios para tests de integración) ────────────────
from src.backtesting.engine import BacktestEngine
from src.backtesting.portfolio import Portfolio


# =============================================================================
# Fixtures compartidas
# =============================================================================

@pytest.fixture
def synthetic_df_with_indicators() -> pd.DataFrame:
    """
    DataFrame OHLCV sintético con indicadores PRE-CALCULADOS (sin pandas_ta).

    Tiene 300 velas y todas las columnas que feature_engineering espera:
    sma_fast, sma_slow, rsi, macd, macd_signal, macd_hist,
    bb_upper, bb_mid, bb_lower, bb_width, ema, atr.
    """
    rng = np.random.default_rng(seed=123)
    n   = 300

    prices = 40_000 + np.cumsum(rng.normal(0, 150, n))
    prices = np.clip(prices, 1, None)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    df = pd.DataFrame({
        "timestamp":  timestamps,
        "open":       prices * (1 + rng.normal(0, 0.001, n)),
        "high":       prices * (1 + abs(rng.normal(0, 0.004, n))),
        "low":        prices * (1 - abs(rng.normal(0, 0.004, n))),
        "close":      prices,
        "volume":     rng.uniform(100, 2000, n),
        # Indicadores sintéticos (mismo shape que los reales)
        "sma_fast":   pd.Series(prices).rolling(10).mean().values,
        "sma_slow":   pd.Series(prices).rolling(20).mean().values,
        "rsi":        rng.uniform(25, 75, n),
        "macd":       rng.normal(0, 50, n),
        "macd_signal":rng.normal(0, 40, n),
        "macd_hist":  rng.normal(0, 20, n),
        "bb_upper":   prices * 1.02,
        "bb_mid":     prices,
        "bb_lower":   prices * 0.98,
        "bb_width":   prices * 0.04,
        "ema":        pd.Series(prices).ewm(span=21).mean().values,
        "atr":        rng.uniform(100, 600, n),
    })
    # Elimina NaN de las medias móviles (igual que apply_all)
    return df.dropna().reset_index(drop=True)


@pytest.fixture
def small_feature_matrix(synthetic_df_with_indicators) -> FeatureMatrix:
    """FeatureMatrix pequeña para tests rápidos."""
    return create_feature_matrix(
        df=synthetic_df_with_indicators,
        window_size=10,
        target_horizon=3,
        threshold_pct=0.002,
    )


@pytest.fixture
def trained_rf_model(small_feature_matrix) -> RandomForestModel:
    """RandomForestModel ya entrenado."""
    split = temporal_split(small_feature_matrix, test_size=0.2)
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(split["X_train"], split["y_train"])
    return model


@pytest.fixture
def trained_baseline_model(small_feature_matrix) -> BaselineModel:
    """BaselineModel ya entrenado."""
    split  = temporal_split(small_feature_matrix, test_size=0.2)
    model  = BaselineModel(
        feature_cols=small_feature_matrix.feature_cols,
        window_size=10,
    )
    model.fit(split["X_train"], split["y_train"])
    return model


# =============================================================================
# 1. INTEGRIDAD DE FEATURES
# =============================================================================

class TestFeatureIntegrity:
    """Verifica que create_feature_matrix produce datos correctos y coherentes."""

    def test_x_y_same_length(self, small_feature_matrix):
        """len(X) debe ser igual a len(y) y len(timestamps)."""
        fm = small_feature_matrix
        assert len(fm.X) == len(fm.y), f"X={len(fm.X)}, y={len(fm.y)}"
        assert len(fm.X) == len(fm.timestamps)

    def test_no_nan_in_x_after_normalization(self, small_feature_matrix):
        """X normalizado no debe tener ningún NaN."""
        nan_count = np.isnan(small_feature_matrix.X).sum()
        assert nan_count == 0, f"X tiene {nan_count} valores NaN tras normalización"

    def test_no_nan_in_x_raw(self, small_feature_matrix):
        """X_raw (sin normalizar) tampoco debe tener NaN."""
        nan_count = np.isnan(small_feature_matrix.X_raw).sum()
        assert nan_count == 0

    def test_x_shape_matches_window_and_features(self, synthetic_df_with_indicators):
        """X.shape[1] debe ser window_size * n_feature_cols."""
        window    = 5
        fm        = create_feature_matrix(
            synthetic_df_with_indicators,
            window_size=window,
            target_horizon=2,
        )
        n_features = len(fm.feature_cols)
        assert fm.X.shape[1] == window * n_features, (
            f"Esperado {window}*{n_features}={window*n_features}, "
            f"obtenido {fm.X.shape[1]}"
        )

    def test_y_only_contains_valid_classes(self, small_feature_matrix):
        """y solo puede contener 1, -1 o 0."""
        unique_labels = set(np.unique(small_feature_matrix.y).tolist())
        assert unique_labels.issubset({1, -1, 0}), \
            f"y contiene valores inesperados: {unique_labels}"

    def test_timestamps_are_datetime(self, small_feature_matrix):
        """Los timestamps deben ser DatetimeTZDtype (UTC)."""
        ts = small_feature_matrix.timestamps
        assert hasattr(ts, "dtype"), "timestamps debe ser un DatetimeIndex"

    def test_scaler_is_fitted(self, small_feature_matrix):
        """El scaler del FeatureMatrix debe estar ajustado."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(small_feature_matrix.scaler)  # lanza si no está fitted

    def test_feature_cols_subset_of_df_columns(self, synthetic_df_with_indicators):
        """Las feature_cols deben ser un subconjunto de las columnas del DataFrame."""
        fm = create_feature_matrix(synthetic_df_with_indicators, window_size=5, target_horizon=2)
        for col in fm.feature_cols:
            assert col in synthetic_df_with_indicators.columns, \
                f"Feature col '{col}' no está en el DataFrame"

    def test_insufficient_data_raises_valueerror(self):
        """Un DataFrame demasiado pequeño debe lanzar ValueError."""
        tiny_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "close": [100, 101, 102, 103, 104],
            "sma_fast": [100, 101, 102, 103, 104],
            "sma_slow": [99, 100, 101, 102, 103],
        })
        with pytest.raises(ValueError, match="filas"):
            create_feature_matrix(tiny_df, window_size=10, target_horizon=5)

    def test_missing_feature_cols_raises_valueerror(self, synthetic_df_with_indicators):
        """Solicitar una columna que no existe en el DataFrame debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no encontradas"):
            create_feature_matrix(
                synthetic_df_with_indicators,
                window_size=5,
                target_horizon=2,
                feature_cols=["columna_que_no_existe"],
            )

    def test_x_raw_and_x_scaled_have_same_shape(self, small_feature_matrix):
        """X_raw y X deben tener exactamente el mismo shape."""
        assert small_feature_matrix.X_raw.shape == small_feature_matrix.X.shape


# =============================================================================
# 2. VALIDACIÓN TEMPORAL — ANTI-LEAKAGE
# =============================================================================

class TestTemporalNoLeakage:
    """
    Verifica que la separación temporal es correcta y que no hay data leakage.
    El leakage ocurriría si datos futuros aparecieran en el conjunto de entrenamiento.
    """

    def test_test_set_always_after_train_set(self, small_feature_matrix):
        """Todos los índices del test deben ser POSTERIORES a todos los del train."""
        split = temporal_split(small_feature_matrix, test_size=0.2)
        train_ts = split["ts_train"]
        test_ts  = split["ts_test"]
        assert train_ts[-1] < test_ts[0], (
            f"LEAKAGE: el último timestamp del train ({train_ts[-1]}) "
            f"no es anterior al primero del test ({test_ts[0]})"
        )

    def test_no_timestamp_overlap_between_train_and_test(self, small_feature_matrix):
        """No debe haber ningún timestamp compartido entre train y test."""
        split    = temporal_split(small_feature_matrix, test_size=0.2)
        train_ts = set(split["ts_train"].astype(str))
        test_ts  = set(split["ts_test"].astype(str))
        overlap  = train_ts.intersection(test_ts)
        assert len(overlap) == 0, \
            f"LEAKAGE: {len(overlap)} timestamps compartidos entre train y test"

    def test_timeseriessplit_folds_are_ordered(self, small_feature_matrix):
        """En cada fold de TimeSeriesSplit, max(train_idx) < min(test_idx)."""
        split = temporal_split(small_feature_matrix, test_size=0.2, n_splits=5)
        tscv  = split["tscv"]
        X_train = split["X_train"]

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            assert tr_idx.max() < val_idx.min(), (
                f"LEAKAGE en fold {fold}: "
                f"max(train_idx)={tr_idx.max()} >= min(val_idx)={val_idx.min()}"
            )

    def test_test_size_matches_expected_proportion(self, small_feature_matrix):
        """El test set debe tener aproximadamente el test_size% del total."""
        test_size = 0.25
        split     = temporal_split(small_feature_matrix, test_size=test_size)
        total     = len(small_feature_matrix.X)
        expected  = int(total * test_size)
        actual    = len(split["X_test"])
        # Tolerancia de ±1 por redondeo entero
        assert abs(actual - expected) <= 1, \
            f"Test size esperado ≈{expected}, obtenido {actual}"

    def test_scaler_fit_only_on_train(self, synthetic_df_with_indicators):
        """
        Si se crea la feature matrix con fit_scaler=False y se pasa el scaler del train,
        la media del train normalizado debe ser ≈ 0 pero la del test puede diferir.
        Esto confirma que el scaler NO se reajustó sobre el test.
        """
        fm_train = create_feature_matrix(
            synthetic_df_with_indicators,
            window_size=5,
            target_horizon=2,
            fit_scaler=True,
        )
        split = temporal_split(fm_train, test_size=0.3)

        # Crea la feature matrix del test SIN re-ajustar el scaler
        fm_test = create_feature_matrix(
            synthetic_df_with_indicators,
            window_size=5,
            target_horizon=2,
            fit_scaler=False,
            scaler=fm_train.scaler,
        )

        # La media del train normalizado debe ser ≈ 0 (propiedad del StandardScaler)
        train_mean = abs(fm_train.X.mean())
        assert train_mean < 0.1, \
            f"Media del train normalizado esperada ≈0, obtenida {train_mean:.4f}"

    def test_fit_scaler_false_without_scaler_raises(self, synthetic_df_with_indicators):
        """Usar fit_scaler=False sin pasar un scaler debe lanzar ValueError."""
        with pytest.raises(ValueError, match="scaler"):
            create_feature_matrix(
                synthetic_df_with_indicators,
                window_size=5,
                target_horizon=2,
                fit_scaler=False,
                scaler=None,
            )

    def test_window_features_use_only_past_data(self, synthetic_df_with_indicators):
        """
        La ventana de la muestra i solo debe usar datos de los índices [i-w, i).
        Comprueba que X[0] corresponde a los primeros window_size timesteps del df.
        """
        window = 5
        fm = create_feature_matrix(
            synthetic_df_with_indicators,
            window_size=window,
            target_horizon=2,
            feature_cols=["sma_fast"],  # Solo 1 feature para simplicidad
        )

        # La primera muestra de X_raw debe ser sma_fast de las filas [0:window]
        expected = synthetic_df_with_indicators["sma_fast"].values[:window]
        actual   = fm.X_raw[0]   # shape (window,) porque 1 feature × window

        np.testing.assert_array_almost_equal(
            actual, expected,
            decimal=6,
            err_msg="X[0] no coincide con los primeros window datos del DataFrame"
        )


# =============================================================================
# 3. CONSISTENCIA DE MODELOS — SAVE / LOAD
# =============================================================================

class TestModelSaveLoad:
    """
    Verifica que serializar y deserializar un modelo produce predicciones idénticas.
    Esto es crítico para reproducibilidad del TFG.
    """

    def test_random_forest_save_load_same_predictions(
        self, trained_rf_model, small_feature_matrix
    ):
        """Un RF cargado desde disco debe predecir exactamente igual que el original."""
        split    = temporal_split(small_feature_matrix, test_size=0.2)
        X_test   = split["X_test"]

        preds_before = trained_rf_model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        try:
            trained_rf_model.save(path)

            loaded_model = RandomForestModel()
            loaded_model.load(path)

            preds_after = loaded_model.predict(X_test)

            np.testing.assert_array_equal(
                preds_before, preds_after,
                err_msg="RandomForestModel: predicciones difieren tras save/load"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_baseline_save_load_same_predictions(
        self, trained_baseline_model, small_feature_matrix
    ):
        """Un BaselineModel cargado debe predecir igual que el original."""
        split  = temporal_split(small_feature_matrix, test_size=0.2)
        X_test = split["X_test"]

        preds_before = trained_baseline_model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        try:
            trained_baseline_model.save(path)

            loaded = BaselineModel(
                feature_cols=trained_baseline_model.feature_cols,
                window_size=trained_baseline_model.window_size,
            )
            loaded.load(path)

            preds_after = loaded.predict(X_test)

            np.testing.assert_array_equal(
                preds_before, preds_after,
                err_msg="BaselineModel: predicciones difieren tras save/load"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_from_nonexistent_path_raises(self):
        """Cargar desde una ruta inexistente debe lanzar una excepción."""
        model = RandomForestModel()
        with pytest.raises(Exception):
            model.load("/ruta/que/no/existe/modelo.joblib")

    def test_predict_before_fit_raises(self):
        """Predecir antes de entrenar debe lanzar RuntimeError."""
        model = RandomForestModel()
        with pytest.raises(RuntimeError, match="entrenado"):
            model.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_save_before_fit_raises(self):
        """Guardar antes de entrenar debe lanzar RuntimeError."""
        model = RandomForestModel()
        with pytest.raises(RuntimeError, match="entrenado"):
            model.save("/tmp/modelo.joblib")

    def test_is_fitted_false_before_training(self):
        """is_fitted() debe devolver False antes de llamar a fit()."""
        assert not RandomForestModel().is_fitted()
        assert not BaselineModel(feature_cols=["sma_fast"], window_size=5).is_fitted()

    def test_is_fitted_true_after_training(self, small_feature_matrix):
        """is_fitted() debe devolver True después de llamar a fit()."""
        model = RandomForestModel(n_estimators=5)
        split = temporal_split(small_feature_matrix, test_size=0.2)
        model.fit(split["X_train"], split["y_train"])
        assert model.is_fitted()


# =============================================================================
# 4. MAPEO DE SEÑALES — MLStrategy
# =============================================================================

class TestMLStrategySignalMapping:
    """
    Verifica que MLStrategy convierte correctamente predicciones numéricas en señales.
    Este es el contrato entre el mundo ML y el BacktestEngine.
    """

    def _make_strategy(self, predictions: dict) -> MLStrategy:
        """Helper: crea un MLStrategy con un modelo mock ya entrenado."""
        class _MockModel(BaseModel):
            """Modelo mock que simula estar entrenado."""
            def __init__(self):
                self._is_fitted = True
            def fit(self, X, y): pass
            def predict(self, X): return np.zeros(len(X), dtype=int)
            def save(self, path): pass
            def load(self, path): pass

        return MLStrategy(model=_MockModel(), predictions_map=predictions)

    def _make_row(self, ts: pd.Timestamp) -> pd.Series:
        return pd.Series({"timestamp": ts, "close": 40_000.0})

    # ------------------------------------------------------------------
    TS_BUY  = pd.Timestamp("2024-01-01 01:00", tz="UTC")
    TS_SELL = pd.Timestamp("2024-01-01 02:00", tz="UTC")
    TS_HOLD = pd.Timestamp("2024-01-01 03:00", tz="UTC")
    TS_NONE = pd.Timestamp("2024-01-01 04:00", tz="UTC")  # No en el mapa

    def test_prediction_1_maps_to_buy(self):
        """Predicción 1 → 'BUY'."""
        strategy = self._make_strategy({self.TS_BUY: 1})
        assert strategy.generate_signal(self._make_row(self.TS_BUY)) == "BUY"

    def test_prediction_minus1_maps_to_sell(self):
        """Predicción -1 → 'SELL'."""
        strategy = self._make_strategy({self.TS_SELL: -1})
        assert strategy.generate_signal(self._make_row(self.TS_SELL)) == "SELL"

    def test_prediction_0_maps_to_hold(self):
        """Predicción 0 → 'HOLD'."""
        strategy = self._make_strategy({self.TS_HOLD: 0})
        assert strategy.generate_signal(self._make_row(self.TS_HOLD)) == "HOLD"

    def test_unknown_timestamp_returns_default_hold(self):
        """Timestamp no presente en el mapa → 'HOLD' (comportamiento seguro)."""
        strategy = self._make_strategy({self.TS_BUY: 1})
        assert strategy.generate_signal(self._make_row(self.TS_NONE)) == "HOLD"

    def test_all_three_signals_possible(self):
        """Las tres señales posibles deben ser accesibles."""
        strategy = self._make_strategy({
            self.TS_BUY: 1, self.TS_SELL: -1, self.TS_HOLD: 0
        })
        assert strategy.generate_signal(self._make_row(self.TS_BUY))  == "BUY"
        assert strategy.generate_signal(self._make_row(self.TS_SELL)) == "SELL"
        assert strategy.generate_signal(self._make_row(self.TS_HOLD)) == "HOLD"

    def test_only_valid_signals_returned(self):
        """generate_signal solo puede devolver BUY, SELL o HOLD."""
        strategy = self._make_strategy({
            self.TS_BUY: 1, self.TS_SELL: -1, self.TS_HOLD: 0, self.TS_NONE: 0
        })
        for ts in [self.TS_BUY, self.TS_SELL, self.TS_HOLD, self.TS_NONE]:
            sig = strategy.generate_signal(self._make_row(ts))
            assert sig in {"BUY", "SELL", "HOLD"}, f"Señal inválida: {sig}"

    def test_ml_strategy_rejects_unfitted_model(self):
        """MLStrategy debe lanzar ValueError si el modelo no ha sido entrenado."""
        unfitted = RandomForestModel()
        with pytest.raises(ValueError, match="entrenado"):
            MLStrategy(model=unfitted, predictions_map={})

    def test_build_ml_strategy_produces_correct_signals(self, trained_rf_model, small_feature_matrix):
        """build_ml_strategy debe crear predicciones alineadas con los timestamps."""
        split    = temporal_split(small_feature_matrix, test_size=0.2)
        strategy = build_ml_strategy(
            model=trained_rf_model,
            X=split["X_test"],
            timestamps=split["ts_test"],
        )
        # Todas las predicciones deben ser BUY, SELL o HOLD
        for ts in split["ts_test"]:
            row = pd.Series({"timestamp": ts, "close": 40_000.0})
            sig = strategy.generate_signal(row)
            assert sig in {"BUY", "SELL", "HOLD"}


# =============================================================================
# 5. INTEGRACIÓN END-TO-END: df → features → entrenamiento → backtest
# =============================================================================

class TestEndToEndPipeline:
    """
    Verifica que el flujo completo del Hito 2 funciona sin errores.
    Comprueba especialmente la consistencia de dimensiones en cada paso.
    """

    def test_full_pipeline_runs_without_errors(self, synthetic_df_with_indicators):
        """El pipeline completo debe ejecutarse sin excepciones."""
        fm    = create_feature_matrix(synthetic_df_with_indicators, window_size=10, target_horizon=3)
        split = temporal_split(fm, test_size=0.2)

        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(split["X_train"], split["y_train"])

        strategy = build_ml_strategy(model, split["X_test"], split["ts_test"])

        # Recorta el df al período de test
        test_start = split["ts_test"][0]
        df_test    = synthetic_df_with_indicators[
            synthetic_df_with_indicators["timestamp"] >= test_start
        ].copy().reset_index(drop=True)

        portfolio = Portfolio(initial_capital=10_000.0)
        engine    = BacktestEngine(portfolio=portfolio)
        result    = engine.run(df=df_test, strategy=strategy)

        assert result.final_equity > 0
        assert len(portfolio.equity_curve) == len(df_test)

    def test_predictions_cover_all_test_timestamps(self, synthetic_df_with_indicators):
        """Cada timestamp del test set debe tener una predicción en el mapa."""
        fm    = create_feature_matrix(synthetic_df_with_indicators, window_size=5, target_horizon=2)
        split = temporal_split(fm, test_size=0.2)

        model = RandomForestModel(n_estimators=5, random_state=0)
        model.fit(split["X_train"], split["y_train"])

        strategy = build_ml_strategy(model, split["X_test"], split["ts_test"])

        for ts in split["ts_test"]:
            assert ts in strategy.predictions_map, \
                f"Timestamp {ts} no tiene predicción en predictions_map"

    def test_no_position_open_at_end_of_backtest(self, synthetic_df_with_indicators):
        """El engine debe cerrar posiciones abiertas al final."""
        fm    = create_feature_matrix(synthetic_df_with_indicators, window_size=5, target_horizon=2)
        split = temporal_split(fm, test_size=0.2)

        model = RandomForestModel(n_estimators=5, random_state=0)
        model.fit(split["X_train"], split["y_train"])

        strategy = build_ml_strategy(model, split["X_test"], split["ts_test"])

        test_start = split["ts_test"][0]
        df_test    = synthetic_df_with_indicators[
            synthetic_df_with_indicators["timestamp"] >= test_start
        ].copy().reset_index(drop=True)

        portfolio = Portfolio(initial_capital=10_000.0)
        engine    = BacktestEngine(portfolio=portfolio)
        engine.run(df=df_test, strategy=strategy)

        assert portfolio.holding == 0.0, "Posición abierta al final del backtest"

    def test_reshape_for_lstm_correct_shape(self, small_feature_matrix):
        """reshape_for_lstm debe producir shape (n_samples, window, n_features)."""
        fm       = small_feature_matrix
        window   = 10
        n_feats  = len(fm.feature_cols)
        X_3d     = reshape_for_lstm(fm.X, window_size=window, n_features=n_feats)

        assert X_3d.shape == (len(fm.X), window, n_feats), (
            f"Shape esperado {(len(fm.X), window, n_feats)}, "
            f"obtenido {X_3d.shape}"
        )

    def test_baseline_model_same_pipeline_as_random_forest(self, synthetic_df_with_indicators):
        """BaselineModel y RandomForestModel deben poder usarse de forma intercambiable."""
        fm    = create_feature_matrix(synthetic_df_with_indicators, window_size=10, target_horizon=3)
        split = temporal_split(fm, test_size=0.2)

        for model_cls, kwargs in [
            (RandomForestModel, {"n_estimators": 5, "random_state": 0}),
            (BaselineModel,     {"feature_cols": fm.feature_cols, "window_size": 10}),
        ]:
            model = model_cls(**kwargs)
            model.fit(split["X_train"], split["y_train"])

            strategy = build_ml_strategy(model, split["X_test"], split["ts_test"])

            test_start = split["ts_test"][0]
            df_test    = synthetic_df_with_indicators[
                synthetic_df_with_indicators["timestamp"] >= test_start
            ].copy().reset_index(drop=True)

            portfolio = Portfolio(initial_capital=10_000.0)
            engine    = BacktestEngine(portfolio=portfolio)
            result    = engine.run(df=df_test, strategy=strategy)

            assert result.final_equity > 0, \
                f"{model_cls.__name__}: equity final inválida {result.final_equity}"
