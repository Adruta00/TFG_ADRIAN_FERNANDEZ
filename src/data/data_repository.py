"""
data_repository.py
==================
Responsabilidad única: persistir y recuperar datos OHLCV desde disco.

Abstrae el formato de almacenamiento (CSV en este hito) del resto del sistema.
Si en el futuro se cambia a SQLite o Parquet, solo se modifica este archivo.

Convención de nombres de archivo:
    {raw_path}/{symbol}_{timeframe}_{start}_{end}.csv
    Ejemplo: data/raw/BTC_USDT_2023-01-01_2024-01-01.csv
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataRepository:
    """
    Gestiona la lectura y escritura de DataFrames OHLCV en disco (CSV).

    Parameters
    ----------
    raw_path : str
        Directorio donde se guardan los datos históricos sin procesar.
    processed_path : str
        Directorio donde se guardan los datos con indicadores calculados.
    """

    def __init__(self, raw_path: str, processed_path: str) -> None:
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)

        # Crea los directorios si no existen
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistencia de datos crudos (OHLCV sin indicadores)
    # ------------------------------------------------------------------

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Guarda un DataFrame OHLCV en CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas: timestamp, open, high, low, close, volume.
        symbol : str   Ej: 'BTC/USDT'
        timeframe : str   Ej: '1h'
        start_date : str  Ej: '2023-01-01'
        end_date : str    Ej: '2024-01-01'

        Returns
        -------
        Path  Ruta del archivo guardado.
        """
        filename = self._build_filename(symbol, timeframe, start_date, end_date)
        filepath = self.raw_path / filename

        df.to_csv(filepath, index=False)
        logger.info("OHLCV guardado → %s (%d filas)", filepath, len(df))
        return filepath

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """
        Carga un DataFrame OHLCV desde CSV si existe en disco.

        Returns
        -------
        pd.DataFrame | None
            DataFrame cargado, o None si el archivo no existe (caché miss).
        """
        filename = self._build_filename(symbol, timeframe, start_date, end_date)
        filepath = self.raw_path / filename

        if not filepath.exists():
            logger.info("Caché miss: %s no encontrado en disco.", filepath)
            return None

        df = pd.read_csv(filepath, parse_dates=["timestamp"])

        # Re-aplica zona horaria UTC (CSV no la preserva)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        logger.info("OHLCV cargado desde caché → %s (%d filas)", filepath, len(df))
        return df

    def ohlcv_exists(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> bool:
        """Comprueba si el CSV ya está descargado (para evitar requests innecesarios)."""
        filename = self._build_filename(symbol, timeframe, start_date, end_date)
        return (self.raw_path / filename).exists()

    # ------------------------------------------------------------------
    # Persistencia de datos procesados (OHLCV + indicadores calculados)
    # ------------------------------------------------------------------

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """Guarda datos con indicadores calculados."""
        filepath = self.processed_path / f"{name}.csv"
        df.to_csv(filepath, index=False)
        logger.info("Datos procesados guardados → %s", filepath)
        return filepath

    def load_processed(self, name: str) -> pd.DataFrame | None:
        """Carga datos con indicadores desde disco."""
        filepath = self.processed_path / f"{name}.csv"
        if not filepath.exists():
            return None
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # ------------------------------------------------------------------
    # Utilidades privadas
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filename(
        symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> str:
        """
        Genera un nombre de archivo determinista.
        'BTC/USDT', '1h', '2023-01-01', '2024-01-01'
            → 'BTC_USDT_1h_2023-01-01_2024-01-01.csv'
        """
        safe_symbol = symbol.replace("/", "_")
        return f"{safe_symbol}_{timeframe}_{start_date}_{end_date}.csv"
