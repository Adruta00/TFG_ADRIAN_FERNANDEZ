"""
test_data.py
============
Tests unitarios para la capa de datos (data_repository.py).

Cubre:
    - Guardado y carga de OHLCV en disco (round-trip)
    - Detección de caché existente
    - Integridad del DataFrame cargado (tipos, columnas, timezone)

NO testea historical_loader.py (requeriría red; se testea manualmente).
"""

import pandas as pd
import pytest

from src.data.data_repository import DataRepository


class TestDataRepository:
    """Suite de tests para DataRepository."""

    @pytest.fixture
    def repo(self, tmp_path) -> DataRepository:
        """Crea un repositorio temporal para cada test (aislado)."""
        return DataRepository(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
        )

    # ------------------------------------------------------------------
    def test_save_and_load_roundtrip(self, repo, sample_ohlcv):
        """Guardar y cargar debe producir el mismo DataFrame."""
        repo.save_ohlcv(sample_ohlcv, "BTC/USDT", "1h", "2024-01-01", "2024-02-01")
        loaded = repo.load_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-02-01")

        assert loaded is not None
        assert len(loaded) == len(sample_ohlcv)

    def test_loaded_columns_match(self, repo, sample_ohlcv):
        """El DataFrame cargado debe tener las mismas columnas que el original."""
        repo.save_ohlcv(sample_ohlcv, "BTC/USDT", "1h", "2024-01-01", "2024-02-01")
        loaded = repo.load_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-02-01")

        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(loaded.columns))

    def test_timestamp_has_utc_timezone(self, repo, sample_ohlcv):
        """El timestamp debe preservar la zona horaria UTC tras cargar desde CSV."""
        repo.save_ohlcv(sample_ohlcv, "BTC/USDT", "1h", "2024-01-01", "2024-02-01")
        loaded = repo.load_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-02-01")

        assert loaded["timestamp"].dt.tz is not None
        assert str(loaded["timestamp"].dt.tz) == "UTC"

    def test_cache_miss_returns_none(self, repo):
        """Cargar un archivo que no existe debe devolver None."""
        result = repo.load_ohlcv("ETH/USDT", "4h", "2023-01-01", "2023-06-01")
        assert result is None

    def test_ohlcv_exists_false_before_save(self, repo):
        """ohlcv_exists() debe devolver False antes de guardar."""
        assert not repo.ohlcv_exists("BTC/USDT", "1h", "2024-01-01", "2024-02-01")

    def test_ohlcv_exists_true_after_save(self, repo, sample_ohlcv):
        """ohlcv_exists() debe devolver True después de guardar."""
        repo.save_ohlcv(sample_ohlcv, "BTC/USDT", "1h", "2024-01-01", "2024-02-01")
        assert repo.ohlcv_exists("BTC/USDT", "1h", "2024-01-01", "2024-02-01")

    def test_numeric_columns_are_float(self, repo, sample_ohlcv):
        """Las columnas OHLCV deben ser numéricas (float64) tras la carga."""
        repo.save_ohlcv(sample_ohlcv, "BTC/USDT", "1h", "2024-01-01", "2024-02-01")
        loaded = repo.load_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-02-01")

        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_float_dtype(loaded[col]), \
                f"Columna '{col}' debería ser float, es {loaded[col].dtype}"

    def test_filename_sanitizes_slash(self, repo, sample_ohlcv):
        """El símbolo 'BTC/USDT' no debe crear subdirectorios (reemplaza '/' por '_')."""
        path = repo.save_ohlcv(sample_ohlcv, "BTC/USDT", "1h", "2024-01-01", "2024-02-01")
        assert "/" not in path.name
        assert "BTC_USDT" in path.name

    def test_save_and_load_processed(self, repo, sample_ohlcv):
        """Datos procesados deben guardarse y recuperarse correctamente."""
        repo.save_processed(sample_ohlcv, "test_processed")
        loaded = repo.load_processed("test_processed")

        assert loaded is not None
        assert len(loaded) == len(sample_ohlcv)

    def test_load_processed_missing_returns_none(self, repo):
        """Cargar datos procesados inexistentes devuelve None."""
        assert repo.load_processed("no_existe") is None
