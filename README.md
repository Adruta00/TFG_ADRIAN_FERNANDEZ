# TFG — Aplicación de Machine Learning al Trading Algorítmico en Criptomonedas

**Autor:** Adrián Fernández de la Rosa  
**Institución:** Universidad Politécnica de Madrid · ETSII  
**Grado:** Ingeniería Informática

---

## Requisitos previos

- Python 3.11 o superior
- Git

---

## ⚙️ Configuración del entorno

### Opción A — `venv` (recomendado para el TFG)

```bash
# 1. Clona el repositorio y entra en el directorio
git clone <URL_DEL_REPO>
cd crypto_trading_tfg

# 2. Crea el entorno virtual
python -m venv .venv

# 3. Actívalo
# En macOS / Linux:
source .venv/bin/activate
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Instala las dependencias
pip install -r requirements.txt

# 5. Verifica la instalación
python -c "import ccxt, pandas, pandas_ta; print('✅ Todo instalado correctamente')"
```

### Opción B — `conda`

```bash
conda create -n tfg python=3.11
conda activate tfg
pip install -r requirements.txt
```

---

## 🚀 Ejecutar el sistema (Hito 1)

```bash
# Modo backtest (descarga datos y ejecuta la estrategia rule-based)
python main.py --mode backtest

# Forzar nueva descarga aunque exista caché
python main.py --mode backtest --force-download

# Usar un archivo de config alternativo
python main.py --mode backtest --config config/config.yaml
```

**Salida esperada:**
```
──────────────────────────────────────────────────
  📊  INFORME DE RESULTADOS — BACKTEST
──────────────────────────────────────────────────
  Capital inicial :      10,000.00 USDT
  Capital final   :      11,243.80 USDT
  PnL absoluto    :      +1,243.80 USDT
  PnL %           :         +12.44%
──────────────────────────────────────────────────
  Sharpe Ratio    :          0.8321
  Max Drawdown    :         -15.23%
  Win Rate        :          58.33%
  Total Trades    :              12
  ...
```

---

## 🧪 Ejecutar los tests

```bash
# Todos los tests
pytest tests/ -v

# Con informe de cobertura
pytest tests/ -v --cov=src --cov-report=term-missing

# Solo un módulo específico
pytest tests/test_metrics.py -v
pytest tests/test_backtesting.py -v

# Solo tests que contengan una palabra clave
pytest tests/ -k "sharpe or drawdown" -v
```

**Resultado esperado:** todos los tests en verde (PASSED).  
No se necesita conexión a internet para los tests (usan datos sintéticos).

---

## 📁 Estructura del proyecto

```
crypto_trading_tfg/
├── config/config.yaml          # ← Edita aquí los parámetros
├── src/
│   ├── data/                   # Descarga y almacenamiento de datos
│   ├── features/               # Indicadores técnicos
│   ├── strategies/             # Estrategias de trading
│   ├── backtesting/            # Motor de backtesting
│   └── evaluation/             # Métricas de rendimiento
├── tests/                      # Tests unitarios (pytest)
├── data/raw/                   # Datos OHLCV descargados (auto-generado)
├── results/                    # Resultados del backtest (auto-generado)
├── main.py                     # Punto de entrada
└── requirements.txt
```

---

## 🗺️ Hoja de Ruta

| Hito | Estado | Descripción |
|------|--------|-------------|
| **Hito 1** | ✅ Completo | Esqueleto funcional: datos → indicadores → backtest → métricas |
| **Hito 2** | ✅ Completo | Random Forest + Baseline como estrategia ML (feature engineering, anti-leakage, save/load) |
| **Hito 3** | ⏳ Pendiente | Modelo LSTM |
| **Hito 4** | ⏳ Pendiente | Simulación en tiempo real (paper trading) |

---

## 📝 Notas para la memoria

- Todos los experimentos se loggean con timestamp en la consola.
- Los trades ejecutados se guardan en `results/trades_history.csv`.
- La curva de equity se guarda en `results/equity_curve.csv`.
- Para cambiar el par o el período, edita únicamente `config/config.yaml`.

---

## 🚀 Ejecutar el sistema (Hito 2)

```bash
# Backtest con Random Forest (entrena y compara vs rule-based)
python main.py --mode backtest --model random_forest

# Backtest con baseline (rule-based envuelto en interfaz ML)
python main.py --mode backtest --model baseline

# Backtest original rule-based (sin cambios del Hito 1)
python main.py --mode backtest --model rule_based

# Opciones combinadas
python main.py --mode backtest --model random_forest --force-download
```

**Salida esperada con `--model random_forest`:**
```
────────────────────────────────────────────────────────────
  HITO 2 — Pipeline ML  |  modelo: random_forest
────────────────────────────────────────────────────────────
  Top 10 features más importantes:
    rsi_t-0                           0.0821
    atr_t-0                           0.0754
    ...

════════════════════════════════════════════════════════════
  📊  COMPARATIVA: RANDOM_FOREST vs RULE-BASED
════════════════════════════════════════════════════════════
  Métrica                   RANDOM_FOREST    RULE_BASED
────────────────────────────────────────────────────────────
  Capital final (USDT)        10,234.56       9,876.43
  PnL %                          +2.35%          -1.24%
  Sharpe Ratio                   0.3421          0.1123
  Max Drawdown                  -8.45%         -12.31%
  Win Rate                       54.17%          41.67%
  Total Trades                       24              12
════════════════════════════════════════════════════════════
```

---

## 🧪 Ejecutar los tests (Hito 2)

```bash
# Solo los tests del Hito 2
pytest tests/test_hito2.py -v

# Todos los tests (Hito 1 + Hito 2)
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=src --cov-report=term-missing
```
