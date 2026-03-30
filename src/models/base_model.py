"""
base_model.py
=============
Define el contrato que TODOS los modelos de ML deben cumplir.

Por qué una clase abstracta:
    - MLStrategy solo necesita saber que cualquier modelo tiene fit/predict.
      No le importa si es Random Forest, LSTM o cualquier otro.
    - Añadir un modelo nuevo = crear un archivo nuevo que herede de BaseModel.
      El resto del sistema (MLStrategy, BacktestEngine, metrics) NO cambia.

Contrato de señales (valores de retorno de predict):
     1  →  BUY
    -1  →  SELL
     0  →  HOLD

IMPORTANTE: Las subclases DEBEN devolver exactamente estos tres valores enteros.
MLStrategy los mapea directamente a las señales "BUY" / "SELL" / "HOLD".
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """
    Clase abstracta que define la interfaz de todos los modelos predictivos.

    Cualquier implementación concreta DEBE implementar los cuatro métodos abstractos.
    """

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entrena el modelo sobre los datos de entrenamiento.

        Parameters
        ----------
        X_train : np.ndarray
            Matriz de features normalizadas. Shape (n_samples, n_features).
            Para LSTM: shape (n_samples, timesteps, n_features) — Hito 3.
        y_train : np.ndarray
            Vector de targets. Shape (n_samples,). Valores: 1, -1, 0.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones para nuevas muestras.

        Parameters
        ----------
        X : np.ndarray
            Matriz de features normalizadas. Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Vector de predicciones. Shape (n_samples,).
            Valores: 1 (BUY), -1 (SELL), 0 (HOLD).
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serializa el modelo entrenado a disco.

        Parameters
        ----------
        path : str
            Ruta del archivo de destino (incluida extensión).
            Convención: .joblib para sklearn, .keras para LSTM.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Carga un modelo previamente serializado desde disco.

        Parameters
        ----------
        path : str
            Ruta del archivo guardado con save().
        """
        ...

    # ------------------------------------------------------------------
    # Métodos concretos (heredados por todas las subclases)
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        """
        Comprueba si el modelo ha sido entrenado.
        Las subclases pueden sobreescribir este método si lo necesitan.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def __repr__(self) -> str:
        status = "entrenado" if self.is_fitted() else "no entrenado"
        return f"{self.__class__.__name__}({status})"
