# Screener financiero EEUU

Proyecto de la asignatura **Proyecto de Ciencia de Datos**. Filtra acciones de NASDAQ y NYSE (unas 1.000 empresas) combinando indicadores técnicos, gestión del riesgo y análisis de sentimiento sobre dos fuentes de texto no estructurado: noticias y presentaciones a la SEC (filings). Los modelos de sentimiento son propios, entrenados sobre Financial PhraseBank: del diccionario a un BERT y un GPT pequeños.

> Las señales de compra y venta son técnicas e informativas, **no una recomendación de inversión**.

## Notebooks, en orden

| Notebook | Qué hace |
|---|---|
| `01_universo_tickers.ipynb` | Lista de empresas de la SEC, filtrada a NASDAQ y NYSE |
| `02_precios_y_recorte.ipynb` | Precios de yfinance y recorte a las 1.000 empresas más líquidas |
| `03_sec_datos_estructurados.ipynb` | Filings, sector Fama-French y cifras financieras de la SEC |
| `04_indicadores_tecnicos.ipynb` | RSI, MACD, Bollinger, Fibonacci, CMF, ADX, EMA, Donchian y ATR |
| `05_senales_y_riesgo.ipynb` | Régimen de mercado, señales, stops, volatilidad, Sharpe y estacionalidad |
| `06_textos.ipynb` | Noticias (FNSPID y NewsAPI) y texto de los filings |
| `07_modelos_sentimiento.ipynb` | Comparación de modelos de sentimiento *(en construcción)* |
| `08_sentimiento_aplicado.ipynb` | Sentimiento de noticias y filings *(pendiente)* |
| `09_backtesting.ipynb` | Simulación histórica y comparación con veto por sentimiento *(pendiente)* |
| `app.py` | Interfaz Streamlit con el screener del día *(pendiente)* |

Las funciones que usan varios notebooks están en `funciones.py`. Las especificaciones completas están en `CLAUDE.md`.

## Cómo ejecutarlo

1. Python 3.12 o superior: `python3 -m pip install -r requirements.txt`
2. Copiar `.env.example` como `.env` y rellenarlo: un User-Agent con contacto real para la SEC y una clave gratuita de [NewsAPI](https://newsapi.org).
3. Ejecutar los notebooks **en orden**, de arriba abajo.

La carpeta `data/` **no se sube al repositorio** (unos 6 GB). Se regenera ejecutando los notebooks en orden: cada uno descarga o calcula lo que necesita y lo guarda en caché en `data/`. El notebook 06 descarga el dataset FNSPID (5,7 GB).

## Datos de terceros

- **FNSPID**: Dong, Fan y Peng (2024), [arXiv 2402.06698](https://arxiv.org/abs/2402.06698). Licencia CC BY-NC 4.0.
- **Financial PhraseBank**: Malo et al. (2014). Licencia CC BY-NC-SA 3.0.
- **SEC EDGAR** y **yfinance** para los datos de empresas y precios.
