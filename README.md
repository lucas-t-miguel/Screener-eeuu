# ¿Mueven las noticias la bolsa? Sentimiento de titulares financieros con cuatro modelos y su relación con las acciones de EEUU

¿Se relaciona el sentimiento de las noticias de una empresa con lo que hace después su acción en bolsa?

Proyecto de la asignatura **Proyecto de Ciencia de Datos**. Se mide el sentimiento de los titulares de unas 1.000 empresas de NASDAQ y NYSE con cuatro modelos, se comprueba cuál lo mide mejor, y se compara la rentabilidad de las acciones, menos la del mercado, después de las noticias positivas, neutrales y negativas.

## Resultados principales

- **Cuatro formas de conseguir un modelo de sentimiento, en el test de Financial PhraseBank** (F1 macro, 968 frases): M1, diccionario Loughran-McDonald, **0,502**; M2, Transformer entrenado solo con las frases etiquetadas, **0,673**; M3, la misma red preentrenada con un corpus propio, **0,586**; **M4, FinBERT preentrenado por otros y ajustado aquí, 0,813**. Decir siempre "neutral" da 0,248. Preentrenar un modelo propio y pequeño empeora (M3 − M2 = −0,087, IC 95 %: −0,13 a −0,05); partir de FinBERT mejora (M4 − M2 = +0,140, IC 95 %: 0,10 a 0,18).
- **Con 300 titulares recientes etiquetados a mano**, fuera del dominio de PhraseBank: M1 **0,460**, M2 **0,376**, M3 **0,375** y **M4 0,626**. M2 y M3 casi se reducen a la mitad; M4 sigue siendo el mejor (M4 − M2 = +0,25, IC 95 %: 0,18 a 0,32).
- **Estudio de eventos** (192.069 titulares de FNSPID, unos 111.000 eventos de empresa y sesión): diferencia de rentabilidad anormal entre noticias positivas y negativas **en la sesión siguiente**: **M4 +0,21 puntos** (IC 95 %: 0,17 a 0,26), **M1 +0,14** (0,10 a 0,19), M2 −0,01 (−0,07 a 0,04) y M3 +0,04 (−0,04 a 0,11). Encuentran relación M4 y M1, y M4 algo más que el diccionario: M4 − M1 = +0,07 puntos (IC 95 %: 0,02 a 0,12, *bootstrap* emparejado por evento). El mismo día de la noticia, la diferencia es mucho mayor (M4 +1,33 puntos), pero está inflada: muchos titulares cuentan un movimiento que ya ha ocurrido.
- **Señales técnicas y noticias** (54.001 señales de compra y venta): las señales técnicas por sí solas no baten al mercado, y **comprar después de noticias negativas sale peor que después de positivas**: +0,31 puntos a 5 sesiones (IC 95 %: 0,10 a 0,51) y +0,49 a 10 (0,23 a 0,76). Las noticias positivas no mejoran una compra frente a no tener noticias, y en las ventas no hay ninguna diferencia concluyente.

## Notebooks, en orden

| Notebook | Pregunta que responde |
|---|---|
| `01_universo_tickers.ipynb` | ¿Qué empresas cotizan en NASDAQ y NYSE, según la SEC? |
| `02_precios_y_recorte.ipynb` | ¿Cuáles son las 1.000 más líquidas, y cómo han cotizado en los últimos 10 años? (precios de yfinance) |
| `03_sec_datos_estructurados.ipynb` | ¿A qué sector pertenece cada empresa (12 sectores de Fama-French) y qué informes ha presentado a la SEC? |
| `04_indicadores_tecnicos.ipynb` | ¿Qué dicen cada día los 5 indicadores técnicos de las señales (RSI, CMF, ADX, cruce de EMA y Donchian)? |
| `05_senales.ipynb` | ¿Cuándo hay una señal técnica de compra o de venta? |
| `06_textos.ipynb` | ¿Qué texto hay de cada empresa? Titulares de FNSPID y NewsAPI, informes de la SEC y el corpus de preentrenamiento |
| `07_modelos_sentimiento.ipynb` | ¿Qué forma de conseguir un modelo mide mejor el sentimiento de un texto financiero? (M1–M4 en Financial PhraseBank) |
| `08_noticias_y_bolsa.ipynb` | ¿Saben los modelos leer titulares reales, y se mueve la acción después de una noticia? |
| `09_senales_y_noticias.ipynb` | ¿Cambia lo que pasa después de una señal técnica según las noticias de los días anteriores? |

Cada notebook se lee de arriba abajo: empieza con su objetivo, sus entradas y sus salidas, cada sección termina con una conclusión, y la última sección, **Comprobaciones**, verifica los resultados con `assert`.

## La app (`app.py`)

Una app de Streamlit para enseñar el proyecto: se abre con `streamlit run app.py`, una vez ejecutados los notebooks. Tiene un menú lateral con cuatro páginas:

- **Proyecto:** la pregunta, los datos y el método en 4 pasos.
- **Modelos de sentimiento:** el F1 macro con su intervalo de los cuatro modelos en el test de PhraseBank y con los 300 titulares a mano, y las diferencias entre modelos.
- **Noticias y bolsa:** la diferencia entre noticias positivas y negativas en la sesión siguiente, por modelo; el mismo día, aparte; y las compras técnicas según las noticias previas.
- **Explorador:** para una empresa y unas fechas (de octubre de 2016 a junio de 2020), las velas con EMA(4) y EMA(8), el canal de Donchian, las señales de compra y venta y, en una franja propia, los días con noticias coloreados según M4 (con los titulares al pasar el ratón); debajo, el RSI y el ADX, y la tabla de titulares.

**La app no recalcula ningún resultado:** los notebooks 07, 08 y 09 guardan sus tablas finales en `data/cache/resultados_*.parquet`, y el Explorador lee los datos diarios que guardaron los notebooks 02, 04, 05 y 08. No carga Keras ni TensorFlow. El tema está en `.streamlit/config.toml`.

## Código compartido y carpetas

- **`app.py`, `paginas/` y `.streamlit/config.toml`:** la app de la sección anterior: la navegación y la plantilla de los gráficos, una página por fichero (con `paginas/estilo.py`, la paleta compartida) y el tema.
- **`funciones.py`:** las funciones que usan varios notebooks: descargas de la SEC, indicadores, señales, el filtro de titulares de cada empresa, la evaluación de los modelos (F1 macro con intervalo por *bootstrap*) y las comparaciones estadísticas.
- **`capas_keras.py`:** las piezas de Keras de los modelos M2, M3 y M4, compartidas por el 07 y el 08. Fija TensorFlow como motor de Keras al importarse.
- **`etiquetas/`:** los 300 titulares etiquetados a mano y la regla con la que se etiquetaron.
- **`resultados/`:** las predicciones del primer M3, guardadas antes de corregir su preentrenamiento (notebook 07, sección 7.4).
- **`data/`** (no está en el repositorio): todo lo descargado o calculado, en caché. `data/cache/` guarda los datos en parquet y `data/modelos/`, los modelos entrenados.

## Cómo ejecutarlo

1. **Python 3.12 o superior** e instalar las dependencias: `python3 -m pip install -r requirements.txt`.
   - `torch` solo se usa una vez, para convertir los pesos de FinBERT, que vienen en el formato de PyTorch. Los modelos se entrenan con Keras sobre TensorFlow.
   - `tensorflow-text` va ligado a la versión de TensorFlow (2.21): si se actualiza una, hay que actualizar la otra.
2. **Copiar `.env.example` como `.env` y rellenarlo:** un User-Agent con un contacto real para la SEC (`SEC_EDGAR_USER_AGENT`) y una clave gratuita de [NewsAPI](https://newsapi.org) (`NEWSAPI_KEY`). La clave de NewsAPI solo hace falta para volver a pedir titulares recientes (`ACTUALIZAR_NEWSAPI = True` en el notebook 06).
3. **Descargar a mano estos cuatro ficheros** y guardarlos donde se indica. Cada notebook comprueba que están y que su huella SHA-256 es la esperada.

| Fichero | Página de descarga | Enlace directo | Dónde guardarlo | Lo usa |
|---|---|---|---|---|
| `Siccodes12.zip` (descomprimido: `Siccodes12.txt`) | [Kenneth French, 12 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html) → *Download industry definitions* | [Siccodes12.zip](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip) | `data/cache/Siccodes12.txt` | 03 |
| `All_external.csv` de FNSPID (5,7 GB) | [Hugging Face, Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) → `Stock_news` | [All_external.csv](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv) | `data/raw/fnspid/All_external.csv` | 06 |
| `FinancialPhraseBank-v1.0.zip` | [Hugging Face, takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) → `data` | [FinancialPhraseBank-v1.0.zip](https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip) | `data/phrasebank/FinancialPhraseBank-v1.0.zip` | 07 |
| `Loughran-McDonald_MasterDictionary_1993-2025.csv` | [SRAF, Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) → enlace *CSV* | [Google Drive](https://drive.google.com/file/d/1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF/view) | `data/loughran_mcdonald/Loughran-McDonald_MasterDictionary_1993-2025.csv` | 07 |

4. **FinBERT se descarga solo** en el notebook 07, de [Hugging Face](https://huggingface.co/yiyanghkust/finbert-pretrain), fijado en el commit `88ab954` y con su huella SHA-256 comprobada.
5. **Ejecutar los notebooks en orden, del 01 al 09**, cada uno de arriba abajo, en JupyterLab (`jupyter lab`) o en VS Code.

**Los gráficos son interactivos (plotly) y no se ven en el visor de GitHub:** hay que abrir los notebooks en Jupyter. Allí se ven sin conexión a internet.

**Cuánto tarda.** La primera vez, lo caro es descargar el texto de los informes de la SEC (06, unas 2 h 30 min, respetando el límite de 10 peticiones por segundo), preentrenar M3 (07, unos 80 min), ajustar M4 (07, unos 25 min) y puntuar con M4 los titulares de FNSPID (08, unos 25 min), todo en el procesador, sin GPU. Cada resultado se guarda en `data/` y, al volver a ejecutar, se lee de disco: con la caché completa, cada notebook tarda de segundos a pocos minutos y no se vuelve a entrenar ni a descargar nada.

**Los titulares de NewsAPI** (notebook 06) solo cubren el último mes, así que se fueron acumulando ejecutando el 06 a menudo, del 19 de octubre al 16 de noviembre de 2026. Los 300 titulares etiquetados a mano salen de esa recogida y están guardados en `etiquetas/`.

## Datos de terceros y licencias

- **FNSPID:** Dong, Fan y Peng (2024), [arXiv 2402.06698](https://arxiv.org/abs/2402.06698). Licencia CC BY-NC 4.0.
- **Financial PhraseBank:** Malo et al. (2014), [arXiv 1307.5336](https://arxiv.org/abs/1307.5336). Licencia CC BY-NC-SA 3.0.
- **Diccionario Loughran-McDonald:** Loughran y McDonald (2011), *Journal of Finance*. Gratuito para investigación académica.
- **Sectores de Fama-French** (12 sectores): [Data Library de Kenneth French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
- **FinBERT** (versión solo preentrenada): Yang, Uy y Huang (2020), [arXiv 2006.08097](https://arxiv.org/abs/2006.08097), y Huang, Wang y Yang (2022), *Contemporary Accounting Research*. Licencia Apache-2.0 ([repositorio yya518/FinBERT](https://github.com/yya518/FinBERT)).
- **SEC EDGAR:** datos públicos de la SEC, con su [política de acceso](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) (User-Agent identificado y como mucho 10 peticiones por segundo).
- **yfinance:** [biblioteca de código abierto](https://github.com/ranaroussi/yfinance) (Apache-2.0) que lee los precios de Yahoo Finance; los datos están sujetos a los términos de uso de Yahoo.

## Limitaciones

- **Sesgo de selección:** las 1.000 empresas se eligieron por su tamaño y liquidez de 2026 y se estudian desde 2016; infla los niveles de rentabilidad, pero mucho menos las comparaciones entre grupos y entre modelos.
- **Sin hora de publicación:** FNSPID solo trae el día de cada noticia, así que una noticia publicada después del cierre se refleja en la sesión siguiente; el estudio no demuestra que se pueda anticipar la bolsa con las noticias.
- **Cobertura:** noticias de 633 empresas, de octubre de 2016 a junio de 2020, y solo titulares, sin el texto del artículo.

> **Aviso:** nada de lo que muestra este proyecto es una recomendación de inversión.
