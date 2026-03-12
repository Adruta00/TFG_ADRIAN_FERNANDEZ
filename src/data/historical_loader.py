"""
historical_loader.py
====================
Responsabilidad única: descargar datos OHLCV históricos desde un exchange
a través de la librería CCXT.

Produce un DataFrame con columnas estandarizadas:
    timestamp (datetime, UTC), open, high, low, close, volume

No escribe a disco ni calcula indicadores: esas responsabilidades pertenecen
a data_repository.py y a technical_indicators.py respectivamente.
"""

import logging
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalLoader:
    """
    Descarga velas OHLCV históricas desde cualquier exchange soportado por CCXT.

    Parameters
    ----------
    exchange_id : str
        Identificador del exchange en CCXT (ej: 'binance', 'kraken').
    symbol : str
        Par de trading (ej: 'BTC/USDT').
    timeframe : str
        Granularidad temporal (ej: '1h', '4h', '1d').
    """

    # Número máximo de velas por request (límite estándar de Binance)
    _BATCH_SIZE = 1000

    def __init__(self, exchange_id: str, symbol: str, timeframe: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe

        # Instancia el exchange CCXT de forma dinámica
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(
            {
                "enableRateLimit": True,  # Respeta los rate limits del exchange
                "timeout": 30000,         # 30 segundos de timeout por request
            }
        )

        # Carga los mercados disponibles (necesario para validar el símbolo)
        self.exchange.load_markets()
        logger.info("Exchange '%s' inicializado. Símbolo: %s | Timeframe: %s",
                    exchange_id, symbol, timeframe)

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Descarga todas las velas OHLCV entre start_date y end_date.

        Gestiona la paginación automáticamente: si el rango solicitado supera
        el límite por request (_BATCH_SIZE), realiza múltiples llamadas y
        concatena los resultados.

        Parameters
        ----------
        start_date : str
            Fecha de inicio en formato 'YYYY-MM-DD' (UTC).
        end_date : str
            Fecha de fin en formato 'YYYY-MM-DD' (UTC).

        Returns
        -------
        pd.DataFrame
            Columnas: timestamp (DatetimeTZDtype UTC), open, high, low, close, volume.
            Ordenado por timestamp ascendente, sin duplicados.

        Raises
        ------
        ccxt.NetworkError
            Si hay problemas de red con el exchange.
        ValueError
            Si el símbolo o timeframe no son soportados por el exchange.
        """
        # Convierte fechas a timestamps en milisegundos (formato CCXT)
        since_ms = self._date_to_ms(start_date)
        until_ms = self._date_to_ms(end_date)

        all_candles: list[list] = []
        current_since = since_ms

        logger.info("Descargando %s %s desde %s hasta %s ...",
                    self.symbol, self.timeframe, start_date, end_date)

        while current_since < until_ms:
            try:
                batch = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_since,
                    limit=self._BATCH_SIZE,
                )
            except ccxt.NetworkError as exc:
                logger.warning("Error de red, reintentando en 5 s: %s", exc)
                time.sleep(5)
                continue

            if not batch:
                break  # No hay más datos disponibles

            # Filtra velas que superan end_date
            batch = [c for c in batch if c[0] < until_ms]

            if not batch:
                break

            all_candles.extend(batch)
            last_ts = batch[-1][0]

            logger.debug("  Lote recibido: %d velas | último timestamp: %s",
                         len(batch), self._ms_to_str(last_ts))

            # Avanza al siguiente lote (última timestamp + 1 ms)
            current_since = last_ts + 1

            # Pausa respetuosa con el rate limit (CCXT ya lo gestiona, pero
            # añadimos un pequeño sleep adicional por seguridad)
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            logger.warning("No se obtuvieron velas para el rango solicitado.")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = self._to_dataframe(all_candles)
        logger.info("Descarga completa: %d velas obtenidas.", len(df))
        return df

    # ------------------------------------------------------------------
    # Métodos privados de utilidad
    # ------------------------------------------------------------------

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        """Convierte 'YYYY-MM-DD' a milisegundos epoch UTC."""
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _ms_to_str(ms: int) -> str:
        """Convierte milisegundos epoch a string legible (para logs)."""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _to_dataframe(candles: list[list]) -> pd.DataFrame:
        """
        Convierte la lista de velas CCXT en un DataFrame limpio.

        Formato CCXT: [timestamp_ms, open, high, low, close, volume]
        """
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Convierte timestamp a datetime con zona horaria UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Elimina duplicados y ordena por tiempo
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df.reset_index(drop=True)

        # Asegura tipos numéricos correctos
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
