# Screener financiero EEUU

Proyecto de la asignatura **Proyecto de Ciencia de Datos**. Filtra acciones de NASDAQ y NYSE (unas 1.000 empresas) combinando indicadores técnicos, gestión del riesgo y análisis de sentimiento sobre dos fuentes de texto no estructurado: noticias y presentaciones a la SEC (filings). Los modelos de sentimiento son propios, entrenados sobre Financial PhraseBank: del diccionario a un BERT y un GPT pequeños.

> Las señales de compra y venta son técnicas e informativas, **no una recomendación de inversión**.

## Notebooks, en orden

| Notebook | Qué hace |
|---|---|
| `01_universo_tickers.ipynb` | Lista de empresas de la SEC, filtrada a NASDAQ y NYSE |
| `02_precios_y_recorte.ipynb` | Precios de yfinance y recorte a las 1.000 empresas más líquidas |
| `03_sec_datos_estructurados.ipynb` | Filings y sector Fama-French de cada empresa, desde la SEC |
| `04_indicadores_tecnicos.ipynb` | RSI, CMF, ADX, EMA y Donchian (los que usan las señales) |
| `05_senales_y_riesgo.ipynb` | Régimen de mercado y señales de compra y venta |
| `06_textos.ipynb` | Noticias (FNSPID y NewsAPI) y texto de los filings |
| `07_modelos_sentimiento.ipynb` | Comparación de modelos de sentimiento *(en construcción)* |
| `08_sentimiento_aplicado.ipynb` | Sentimiento de noticias y filings *(pendiente)* |
| `09_backtesting.ipynb` | Simulación histórica y comparación con veto por sentimiento *(pendiente)* |
| `app.py` | Interfaz Streamlit con el screener del día *(pendiente)* |

Las funciones que usan varios notebooks están en `funciones.py`.

## Cómo ejecutarlo

1. Python 3.12 o superior: `python3 -m pip install -r requirements.txt`
2. Copiar `.env.example` como `.env` y rellenarlo: un User-Agent con contacto real para la SEC y una clave gratuita de [NewsAPI](https://newsapi.org).
3. Descargar a mano los cuatro ficheros de la tabla y guardarlos donde indica. Cada notebook comprueba que están y que su huella SHA-256 es la esperada.
4. Ejecutar los notebooks **en orden**, de arriba abajo.

| Fichero | Página de descarga | Enlace directo | Dónde guardarlo | Lo usa |
|---|---|---|---|---|
| `Siccodes12.zip` (descomprimido: `Siccodes12.txt`) | [Kenneth French, 12 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html) → *Download industry definitions* | [Siccodes12.zip](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip) | `data/cache/Siccodes12.txt` | 03 |
| `All_external.csv` de FNSPID (5,7 GB) | [Hugging Face, Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) → `Stock_news` | [All_external.csv](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv) | `data/raw/fnspid/All_external.csv` | 06 |
| `FinancialPhraseBank-v1.0.zip` | [Hugging Face, takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) → `data` | [FinancialPhraseBank-v1.0.zip](https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip) | `data/phrasebank/FinancialPhraseBank-v1.0.zip` | 07 |
| `Loughran-McDonald_MasterDictionary_1993-2025.csv` | [SRAF, Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) → enlace *CSV* | [Google Drive](https://drive.google.com/file/d/1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF/view) | `data/loughran_mcdonald/Loughran-McDonald_MasterDictionary_1993-2025.csv` | 07 |

La carpeta `data/` **no se sube al repositorio** (unos 6 GB). Salvo esos cuatro ficheros, se regenera ejecutando los notebooks en orden: cada uno descarga de la SEC, yfinance o NewsAPI, o calcula, lo que necesita y lo guarda en caché en `data/`.

## Datos de terceros

- **FNSPID**: Dong, Fan y Peng (2024), [arXiv 2402.06698](https://arxiv.org/abs/2402.06698). Licencia CC BY-NC 4.0.
- **Financial PhraseBank**: Malo et al. (2014). Licencia CC BY-NC-SA 3.0.
- **Diccionario Loughran-McDonald**: Loughran y McDonald (2011), *Journal of Finance*. Gratuito para investigación académica.
- **Sectores de Fama-French** (12 sectores): Data Library de Kenneth French.
- **SEC EDGAR** y **yfinance** para los datos de empresas y precios.
