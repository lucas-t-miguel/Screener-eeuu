# CLAUDE.md — Screener Financiero EEUU (Proyecto de Ciencia de Datos)

Debe leerse íntegramente antes de generar o modificar código.

> **Última revisión: 19 oct 2026.** Cambios del 19 oct: sentimiento con **modelos propios** entrenados sobre Financial PhraseBank (comparación central BERT propio vs GPT propio); noticias de FNSPID (2016–2020) y NewsAPI (último mes); plan reducido a **4 notebooks** (06–09) y **dos comparaciones**, cortas. Cambios del 11 oct: (1) el proyecto pasa de módulos `.py` a **notebooks Jupyter** con un único `funciones.py` compartido; se eliminan `config/`, `src/`, `tests/` y el logger; (2) se amplía el alcance con la metodología del proyecto previo **Trade Recommender** (ADX, CMF, cruce de EMA, Donchian, trailing-stop, backtesting); (3) se añaden los criterios de evaluación de la asignatura y los experimentos de comparación. El razonamiento de cada cambio está en la bitácora (sección 10).

---

## 0. Contexto y relación con el proyecto anterior

Este proyecto **empieza de cero** como práctica de la asignatura **Proyecto de Ciencia de Datos**. Existe un proyecto personal previo, **Trade Recommender**, más amplio y multi-mercado, que ya validó en la práctica gran parte del enfoque (indicadores técnicos, régimen de mercado, gestión de riesgo con trailing-stop, backtesting, y la ausencia de ventaja predictiva de un modelo ML frente a reglas explícitas). Ese proyecto es **referencia de que el enfoque funciona y fuente de los parámetros**, no código a copiar: aquí todo se reescribe de forma más simple y legible.

**Diferencias clave respecto al proyecto anterior:**

1. **Alcance de mercado:** solo acciones de EEUU (NASDAQ/NYSE). Sin mapeo multi-sufijo (`.MC`, `.OL`, etc.).
2. **Nueva fuente de datos: SEC EDGAR (sec.gov)**, con APIs oficiales (datos estructurados) y descarga/parseo de documentos (texto narrativo) — ver 8.7.
3. **El análisis de sentimiento se completa aquí** (quedó pendiente en Trade Recommender), con **dos fuentes de texto no estructurado**: noticias y presentaciones regulatorias de la SEC.
4. **Formato académico:** notebooks legibles de arriba abajo en lugar de una arquitectura de paquetes Python.
5. **Documentación en paralelo obligatoria** en la bitácora (sección 10), para ensamblar la memoria final.

---

## 1. Identidad y rol del agente

Actúas como **ingeniero de software senior especializado en sistemas cuantitativos y ciencia de datos en Python**, colaborando con un estudiante de Ciencia de Datos e IA. El proyecto tiene triple propósito:

1. **Académico (prioritario):** demostrar un pipeline de datos completo y metodológicamente honesto: ingesta de fuentes heterogéneas, tratamiento de datos no estructurados (texto), feature engineering, comparación de modelos/procedimientos y evaluación crítica de resultados — incluidos resultados nulos.
2. **Formativo / portfolio:** el usuario debe **entender y poder defender cada línea** ante el tribunal. Ante la duda entre una solución sofisticada y una legible, elige la legible.
3. **Funcional:** herramienta real de análisis y filtrado de acciones.

---

## 2. Criterios de evaluación de la asignatura

El profesor **no evalúa el rendimiento económico** de la estrategia, sino **cómo se tratan los datos**. Pide explícitamente:

- **Tratamiento de datos no estructurados** (requisito obligatorio → el módulo de sentimiento sobre noticias y filings SEC es la columna vertebral del proyecto, no un extra).
- **Comparaciones entre modelos o procedimientos** (le gustan especialmente → sección 8.9).

Se evalúa: dificultad, originalidad, desarrollo de la práctica en clase a lo largo del curso, calidad del artículo que la documenta, resultados obtenidos, preparación de la presentación, ejecución de la presentación y adecuación de las respuestas a las preguntas.

**Implicación para el agente:** prioriza siempre lo que alimenta estos criterios (rigor en el tratamiento del texto, comparaciones bien diseñadas, resultados honestos y explicables) sobre la optimización de la rentabilidad.

---

## 3. Descripción del proyecto

**Objetivo:** plataforma que filtra acciones del mercado estadounidense combinando indicadores técnicos, gestión de riesgo y análisis de sentimiento con IA sobre dos fuentes de texto (noticias y filings SEC).

**Alcance del análisis:** señales de **compra y de venta** (simétricas), con niveles de stop y rango de precio de referencia calculados con indicadores técnicos objetivos — **nunca con modelos predictivos de caja negra** presentados como predicción de precio.

**Honestidad metodológica:** si una hipótesis (p. ej. "el sentimiento mejora la señal técnica") da un resultado nulo o débil, se documenta tal cual. Un hallazgo negativo bien medido es un resultado válido.

---

## 4. Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12+ (lo exige `pandas-ta==0.4.71b0`; instalado: 3.13) |
| Entorno de trabajo | Jupyter notebooks (`.ipynb`) + un `funciones.py` compartido |
| Datos de mercado | `yfinance` |
| Análisis técnico | `pandas`, `pandas-ta` |
| Visualización | `plotly` |
| Datos SEC | APIs oficiales EDGAR (`data.sec.gov/submissions`, `data.sec.gov/api/xbrl/frames`) + `requests` y `BeautifulSoup` para el texto narrativo de los filings |
| Noticias | FNSPID (dataset público, titulares 2016–2020) y NewsAPI (plan gratuito, último mes) |
| Sentimiento (PLN) | **Modelos propios** con `keras` 3 sobre el motor **PyTorch** (`torch`, con la GPU del Mac por MPS), entrenados sobre Financial PhraseBank (8.6). Tokenizador WordPiece con `tokenizers`. Sin `keras-hub`: el bloque Transformer, el embedding posicional y las máscaras se escriben a mano con capas de `keras`, para poder explicarlos. Ni Claude ni otro LLM por API. FinBERT (`transformers`) solo como techo opcional |
| Comparación de modelos | `scikit-learn` (TF-IDF, partición, métricas) |
| Interfaz | Streamlit (≥ 1.66, compatible con el `protobuf` ≥ 6 que exige TensorFlow 2.21), en un único `app.py` construido al final |
| Claves | `.env` + `python-dotenv` |

> **Sin entorno virtual:** por decisión consciente del usuario, las dependencias se instalan en el Python global (`python3 -m pip install -r requirements.txt`). Documentado en la bitácora; reversible en cualquier momento porque `requirements.txt` está completo.

> **Motor de Keras: PyTorch, fijado en cada notebook.** Keras 3 funciona sobre TensorFlow o PyTorch con el mismo código. Se usa PyTorch porque en el Mac del autor (Apple M5 Pro) entrena unas 3,3 veces más rápido: 51.600 frente a 15.500 tokens/s, medido el 20 oct 2026 con un Transformer de 11 M de parámetros, porque usa la GPU del Mac y TensorFlow solo el procesador. Todos los modelos usan el **mismo motor**, para que la comparación sea justa. Cada notebook con Keras lo fija en su primera celda de código, **antes de importar Keras**: `os.environ["KERAS_BACKEND"] = "torch"`. Así no depende del `KERAS_BACKEND=torch` del `~/.zshrc` del autor (que es de otros proyectos y no se toca) ni de cómo arranque el kernel en VS Code. Para cambiar de motor basta con cambiar esa línea y reiniciar el kernel. El código usa solo Keras (`keras.layers`, `keras.ops`), nada propio de TensorFlow (`tf.data`, `tf.strings`) ni de PyTorch, para que siga funcionando con cualquiera de los dos.

> **Apuntes de clase en TensorFlow 2.13 / Keras 2:** se adaptan a Keras 3 y pandas 2 (p. ej. `DataFrame.append` → `pd.concat`). Cada adaptación se anota en la bitácora.

> **Sin Selenium ni Playwright:** los filings de EDGAR (10-K, 10-Q, 8-K) son HTML estático; `requests` + `BeautifulSoup` basta.

---

## 5. Reglas operativas de desarrollo (obligatorias)

1. **Modular e iterativo.** Nunca generar el proyecto completo ni un bloque entero en una sola respuesta. Un paso = un notebook, o una sección de un notebook.
2. **De lo básico a lo complejo.** Cada notebook debe funcionar y verificarse antes de pasar al siguiente.
3. **Un paso a la vez.** Al terminar cada entregable, explicar qué se ha construido y cómo comprobarlo, y **esperar confirmación** antes de seguir.
4. **Nunca asumir el formato de datos reales.** Antes de escribir un parser, mirar la respuesta real de la fuente (SEC, yfinance, NewsAPI, un filing concreto) y mostrarla en el notebook.
5. **Pocos notebooks, cada uno con un tema.** Un notebook puede tener varias etapas del mismo tema (p. ej. las dos fuentes de texto), cada una en su sección con la pregunta que responde y una frase de conclusión. Las funciones que se usan en más de un sitio viven en `funciones.py`, **nunca copiadas** entre notebooks: el screener, el backtesting y `app.py` deben usar exactamente el mismo cálculo.
6. **Sin modelos de caja negra para precios.** Stops y rangos de precio salen de indicadores verificables (ATR, Fibonacci, Bollinger). Un modelo estadístico/ML solo se usa para **evaluar** si algo aporta valor (sección 8.9), nunca para generar el nivel de precio mostrado al usuario.
7. **Errores y límites de API explícitos.** Toda llamada externa contempla timeouts, rate limits y datos faltantes, y falla con un mensaje claro, nunca en silencio. SEC EDGAR: `User-Agent` con contacto real y máximo ~10 peticiones/segundo.
8. **Disclaimer permanente.** Cualquier salida de compra/venta es una señal técnica informativa, no una recomendación de inversión.
9. **Documentar decisión y razonamiento en el momento**, en la bitácora (sección 10).
10. **Legibilidad por encima de todo.** Nombres de variables y funciones en español y descriptivos, celdas cortas, nada de abstracciones innecesarias (clases, decoradores, patrones de diseño) si una función simple basta. Si el usuario no puede explicar una celda, la celda está mal escrita.

---

## 6. Estructura del proyecto

```
PCD/
├── .env                          # claves reales (NO se sube a git)
├── .env.example                  # plantilla sin valores
├── .gitignore
├── requirements.txt
├── CLAUDE.md
├── funciones.py                  # funciones reutilizables (indicadores, señales, stops, llamadas a APIs)
├── 01_universo_tickers.ipynb     # SEC: lista de empresas → NASDAQ + NYSE
├── 02_precios_y_recorte.ipynb    # yfinance: precios OHLCV + recorte a ~1.000 tickers
├── 03_sec_datos_estructurados.ipynb  # SEC: submissions (filings + sector) + cifras financieras (frames)
├── 04_indicadores_tecnicos.ipynb # RSI, MACD, Bollinger, Fibonacci, CMF, ADX, EMA, Donchian, ATR
├── 05_senales_y_riesgo.ipynb     # régimen de mercado, señales compra/venta, stops, Sharpe, estacionalidad
├── 06_textos.ipynb               # texto no estructurado: noticias (FNSPID + NewsAPI) y filings SEC; limpieza y corpus
├── 07_modelos_sentimiento.ipynb  # PhraseBank, tokenizador, M0, M1, M2, M3a, M3b: comparación de modelos
├── 08_sentimiento_aplicado.ipynb # el mejor modelo sobre noticias y filings, ~200 titulares a mano, comparación descriptiva
├── 09_backtesting.ipynb          # simulación histórica, cartera y comparación reglas / veto noticias / veto filings
├── app.py                        # interfaz Streamlit con el screener del día (último paso)
├── archivo/                      # versiones completas retiradas de la secuencia principal
└── data/
    ├── cache/                    # todo lo descargado o calculado, en parquet
    ├── raw/fnspid/               # FNSPID original (All_external.csv, 5,7 GB; no se sube a git)
    ├── phrasebank/               # Financial PhraseBank (CC BY-NC-SA 3.0, Malo et al. 2014)
    └── modelos/                  # pesos de los modelos entrenados (.keras) y tokenizador
```

Los notebooks van en la raíz junto a `funciones.py` para que `from funciones import ...` funcione sin trucos de rutas.

---

## 7. Convenciones de los notebooks

- **Primera celda (markdown):** título, objetivo del notebook en 2-3 frases, **entradas** (archivos de `data/` que lee) y **salidas** (archivos que escribe).
- **Segunda celda (código):** imports y `load_dotenv()`. Nada más.
- **Antes de cada bloque de código, una celda markdown** que explique en lenguaje llano qué se hace y por qué. Pensadas para leerse en la defensa.
- **Independencia entre notebooks:** cada notebook lee sus entradas de `data/` y guarda sus salidas en `data/` (parquet). Nunca depende de variables que quedaron en memoria de otro notebook.
- **Caché obligatoria** para todo lo externo o caro (SEC, yfinance, fuentes de noticias, entrenamiento y predicciones de los modelos de sentimiento): si el archivo ya existe en `data/cache/` (o `data/modelos/`), se lee de disco en vez de repetir la llamada o el entrenamiento. Re-ejecutar un notebook no puede volver a entrenar un modelo de una hora ni agotar el cupo diario de una API.
- **Semilla fija** (`SEMILLA = 42` en `funciones.py`) para particiones, inicialización de redes y muestreos, para que *Restart & Run All* dé las mismas cifras.
- **Sección final "Comprobaciones":** celdas con `assert` que verifican los resultados (sin valores nulos donde no debe haberlos, RSI entre 0 y 100, número de filas esperado, etc.). Sustituyen a `tests/` y pytest.
- **Cierre de un notebook:** solo se da por terminado tras ejecutar *Restart & Run All* sin errores. Se guarda **con las salidas visibles**, para que el profesor vea resultados sin ejecutarlo.
- `print()` y `display()` están permitidos; no se usa ningún sistema de logging.
- Gráficos con `plotly`.

**Reglas de `funciones.py`:**

- Solo definiciones de funciones y constantes; ningún código que se ejecute al importarlo.
- Cada función con type hints y un docstring corto en español (qué hace, qué recibe, qué devuelve).
- Una función entra en `funciones.py` cuando se usa en más de un sitio (dos notebooks, o un notebook y `app.py`). Si solo se usa en un notebook, se define en ese notebook.
- Ante datos inválidos, `raise` con un mensaje claro, nunca devolver un resultado vacío en silencio.

---

## 8. Especificaciones funcionales

### 8.1 Universo de tickers (notebooks 01 y 02)

- **Fuente:** `https://www.sec.gov/files/company_tickers_exchange.json` (formato `{"fields": [...], "data": [[...]]}` con `cik`, `name`, `ticker`, `exchange`). No figura en la documentación oficial de la SEC, a diferencia de `company_tickers.json`, pero es un endpoint real de sec.gov y contiene exactamente los mismos CIK que el fichero documentado (8.017 de 8.017, 100 %, verificado el 11 oct 2026 en el notebook 01); se usa porque es el único que trae `exchange`. Decisión y validación cruzada documentadas en la bitácora.
- **Filtro:** NASDAQ + NYSE → 7.659 tickers (verificado: 4.359 Nasdaq + 3.300 NYSE, sobre 10.422 entradas).
- **Recorte a ~1.000 tickers (notebook 02):** primero se descartan las empresas por debajo de un **suelo de capitalización**; después se ordena el resto por **volumen medio en dólares** (cierre × volumen) y se quedan las ~1.000 más líquidas. El umbral de capitalización y la ventana del volumen medio (sugerencia: 63 sesiones) se fijan **viendo la distribución real** en el notebook y se documentan en la bitácora. Antes de pedir la capitalización de 7.659 tickers a yfinance, verificar su coste y límites (regla 4); si es lento, pedirla solo a los tickers que ya superen un mínimo de volumen.

### 8.2 Indicadores técnicos (notebook 04)

Calculados sobre el histórico diario OHLCV de cada acción.

| Indicador | Parámetros | Uso en el sistema |
|---|---|---|
| RSI | 14 sesiones; umbrales 30 / 70 | Señal en régimen lateral (8.3) |
| MACD | 12 / 26 / 9 | Contexto visual en la vista de detalle |
| Bandas de Bollinger | 20 sesiones, ±2σ | Contexto visual + rango de precio esperado |
| Fibonacci dinámico | máx./mín. de 45 sesiones; 23,6 / 38,2 / 50 / 61,8 / 78,6 % | Rango de precio esperado |
| CMF (Chaikin Money Flow) | 20 sesiones; umbral ±0,07 | Confirmación por volumen del cruce de EMA |
| ADX | 14 sesiones; umbral 25 | Clasificación de régimen de mercado |
| Cruce de EMA | EMA(4) vs EMA(8) | Señal en régimen de tendencia |
| Canal de Donchian | 20 sesiones | Señal de ruptura en cualquier régimen |
| ATR | 14 sesiones | Base de todos los stops |

Se usa `pandas-ta` para los cálculos. Si algún indicador se implementa a mano para que se entienda mejor, la sección de Comprobaciones lo verifica contra `pandas-ta`.

### 8.3 Régimen de mercado y señales (notebook 05)

**Régimen:** cada día y cada activo se clasifica con el ADX: **tendencia** si ADX ≥ 25, **lateral** si ADX < 25.

**Tres mecanismos de entrada independientes** (la venta es siempre el espejo de la compra):

1. **Cruce de EMA confirmado** (solo en tendencia): EMA(4) cruza por encima de EMA(8) **y** ADX ≥ 25 **y** CMF ≥ 0,07. Las tres condiciones a la vez.
2. **Reversión por RSI** (solo en lateral): el RSI estaba por debajo de 30 y hoy vuelve a cruzarlo hacia arriba, **y** ADX < 25.
3. **Ruptura de Donchian** (cualquier régimen): el cierre de hoy supera el máximo de las 20 sesiones anteriores.

**Regla de combinación:** hay señal si se cumple (1) **o** (2) **o** (3).

**Condiciones previas a cualquier señal** (fijadas en el notebook 05, 11 oct 2026):

- **Calentamiento:** las primeras 60 sesiones del histórico de cada empresa no tienen régimen ni señales. El ADX de `pandas-ta` sale inflado al principio (media ~48 en las 30 primeras sesiones frente a ~24 después) y no se estabiliza hasta ~60 sesiones.
- **Liquidez mínima:** solo hay señal si la mediana del volumen en dólares (cierre × volumen) de las últimas 20 sesiones, hasta el día de la señal incluido, es de al menos **1 M USD/día**. Excluye las etapas inoperables de empresas que cotizaban sobre todo fuera de EEUU o eran diminutas (1,6 % de los empresa-día). Los precios no se modifican.
- Un día que cumpla a la vez compra y venta se marca como **conflicto** y no genera operación (0 casos en los datos).

### 8.4 Riesgo y gestión de la operación (notebook 05)

- **ATR de entrada:** se calcula el día de la entrada y se congela durante toda la operación.
- **Stop-loss fijo:** 2 × ATR por debajo del precio de entrada (por encima en una venta).
- **Trailing-stop (Chandelier Exit):** máximo alcanzado desde la entrada − 3 × ATR (en venta, mínimo + 3 × ATR).
- **Stop activo:** en cada momento, el más protector de los dos.
- **Sin take-profit fijo.** En Trade Recommender, fijar el objetivo en un soporte/resistencia cercano cerraba la mayoría de operaciones ganadoras al día siguiente. El trailing-stop es el único mecanismo que cierra una operación ganadora.
- **Volatilidad histórica anualizada:** desviación estándar de los retornos diarios × √252.
- **Sharpe Ratio:** retorno medio / volatilidad, anualizado, con tasa libre de riesgo = 0 %.
- **Rango de "precio esperado" (solo informativo):** soporte/resistencia más cercanos combinando Fibonacci y Bollinger; si no hay nivel cercano por un lado, 2 × ATR. **No** decide el cierre de ninguna operación.

### 8.5 Estacionalidad (notebook 05, informativa)

- Retorno medio y win rate (% de años en positivo) **por mes del año**; retorno medio **por día de la semana**.
- Histórico de ~5 años; cada cifra muestra siempre su tamaño de muestra.
- No participa en la señal de compra/venta.

### 8.6 Análisis de sentimiento — dos fuentes de texto no estructurado (notebooks 06, 07 y 08)

**Fuentes de texto (notebook 06)**

- **Noticias:**
  - **FNSPID** (Dong, Fan y Peng 2024, arXiv 2402.06698; CC BY-NC 4.0, uso académico citando): titulares con el ticker en cada fila. Para nuestro universo, 464.983 titulares sin duplicados de 639 de las 1.000 empresas, de octubre de 2016 a junio de 2020; el 41 % habla de su propia empresa (633 empresas con alguno). Solo titular, sin texto del artículo.
  - **A qué empresa se refiere un titular** (`menciona_empresa` en `funciones.py`, revisada el 22 oct 2026): ticker en mayúsculas y suelto (los de una letra o ambiguos, solo entre paréntesis o tras la bolsa) o palabra clave del nombre, con 87 palabras clave revisadas a mano (`PALABRA_CLAVE_A_MANO`): las compartidas por dos empresas, las más frecuentes en los titulares y las que dieron falsos positivos.
  - **NewsAPI** (plan gratuito: 100 peticiones/día, último mes, máximo 100 resultados por consulta): titular y resumen corto de las noticias recientes, para el screener del día.
- **Filings SEC:** Item 1A (riesgos) y Item 7 (MD&A) del 10-K/10-Q, y cuerpo de los 8-K. Las ~110 empresas con 20-F/40-F y XOM (CIK nuevo sin histórico) se deciden en el notebook 06.
- **Corpus de preentrenamiento** de M3a/M3b: texto de los filings más los titulares de noticias.
- **Agregación:** por empresa. El sector Fama-French del notebook 03 se usa solo para desglosar resultados.

**Modelos (notebook 07)**

- **Etiquetas:** Financial PhraseBank (Malo et al. 2014; ~4.840 frases; negativo/neutral/positivo; CC BY-NC-SA 3.0, uso académico citando el artículo). Se entrena con **50Agree** y se reporta también **AllAgree** como sensibilidad.
- **Partición estratificada 60/20/20** (entrenamiento/validación/test) con semilla fija, **la misma para todos los modelos**. La validación elige épocas e hiperparámetros; el test se mira una sola vez al final. (Los apuntes de clase validan sobre el test: es una fuga de información y aquí no se hace.)
- **Métricas:** F1 macro (principal, porque neutral es la clase mayoritaria), exactitud y matriz de confusión. Referencia mínima: predecir siempre la clase mayoritaria.
- **Escalera de modelos:**

| Modelo | Qué es | Preentrenamiento |
|---|---|---|
| M0 | Diccionario Loughran-McDonald | ninguno (sin entrenamiento) |
| M1 | TF-IDF con bigramas + red densa | ninguno |
| M2 | BERT propio **sin preentrenar** (Transformer encoder con embedding posicional) | ninguno |
| M3a | El mismo BERT propio, preentrenado con máscaras (MLM) | corpus propio |
| M3b | GPT propio, preentrenado por siguiente token; clasifica con el vector del último token real | el mismo corpus |

- **M2 frente a M3a** mide cuánto aporta preentrenar (misma red). **M3a frente a M3b** es la comparación central: encoder frente a decoder, con el mismo corpus, el mismo tokenizador y un número de parámetros parecido. Se reporta también la perplejidad del GPT.
- **Tokenizador WordPiece común** a M2, M3a y M3b (librería `tokenizers`), entrenado sobre el corpus propio.
- **Tamaño contenido:** modelos pequeños que se entrenen en minutos en el Mac del autor; una sola tabla final con todas las métricas.

**Aplicación (notebook 08)**

- El mejor modelo puntúa noticias y filings. **Validación fuera de dominio:** ~200 titulares de nuestras noticias etiquetados a mano por el autor (plantilla en este notebook).
- Salida, JSON puro sin texto adicional:

```json
{
  "fuente": "noticia" | "filing_sec",
  "sentimiento": -1 | 0 | 1,
  "catalizador": "el propio titular o frase clasificada",
  "confianza": 0.0-1.0
}
```

`confianza` es la probabilidad que el modelo asigna a la clase elegida. `catalizador` es el texto de entrada (un clasificador no genera texto).

**Uso y coste**

- **Uso en el screener:** filtro/veto de riesgo (p. ej. marcar con precaución o vetar una compra técnica si hay un catalizador negativo fuerte y reciente), no un generador de señales propio.
- **Coste:** los modelos son locales y sin coste por llamada, así que se puntúa todo el histórico. Las predicciones se cachean en disco.
- **Asimetría de histórico:** noticias de oct 2016 a jun 2020 (FNSPID) y del último mes (NewsAPI); filings de todo el periodo. Las comparaciones con noticias usan el periodo común y la limitación se documenta junto a sus resultados.

### 8.7 Ingesta desde SEC EDGAR — dos niveles (notebooks 01, 03 y 06)

1. **Datos estructurados (API oficial, sin scraping; notebook 03):**
   - `data.sec.gov/submissions/CIK##########.json` → lista de filings (tipo, fecha, número de acceso, documento principal) y código SIC de la empresa, que se traduce a los 12 sectores de Fama-French (tabla publicada por Kenneth French).
   - `data.sec.gov/api/xbrl/frames/us-gaap/{etiqueta}/USD/{periodo}.json` → una cifra financiera de todas las empresas en un periodo (ingresos, beneficio neto, activos y patrimonio de 2025). Se usa en lugar de `companyfacts` (3,8 MB por empresa, ~4 GB para las 1.000) porque solo se necesitan unas pocas cifras de contexto. No cubre a las extranjeras que presentan con IFRS.
2. **Texto narrativo (descarga dirigida + parseo; notebook 06):** localizar el documento principal a partir del punto 1, descargarlo con `requests`, extraer las secciones de interés con `BeautifulSoup`. Ese texto es la entrada del sentimiento `filing_sec`.
3. **Política de acceso:** no requiere cuenta ni API key. `User-Agent` con contacto real (variable `SEC_EDGAR_USER_AGENT` del `.env`) y máximo ~10 peticiones/segundo.

### 8.8 Backtesting (notebook 09)

**Reglas de la simulación:**

- Una sola posición abierta por activo a la vez.
- **Entrada a la apertura de la sesión siguiente a la señal** (corrige una limitación documentada de Trade Recommender, que entraba al cierre del mismo día).
- Coste de 0,1 % por entrada y 0,1 % por salida.
- Salida por el stop activo (8.4) o, como máximo, a las 60 sesiones.
- Hasta 10 años de histórico por activo.

**Cartera con capital compartido:** riesgo del 1 % del capital por operación (tamaño calculado con la distancia real al stop), máximo 20 % del capital en una sola posición.

**Métricas:** % de operaciones ganadoras, profit factor, duración media, CAGR y drawdown máximo, separadas para compra y venta.

**Limitaciones que deben documentarse junto a los resultados:**

- **Sesgo de supervivencia:** la lista actual de la SEC solo incluye empresas que siguen cotizando hoy.
- Las ventas en corto no modelan coste ni disponibilidad de préstamo de acciones.
- El sentimiento de noticias solo puede simularse de oct 2016 a jun 2020 y en las 633 empresas con titulares sobre sí mismas (8.6).

### 8.9 Comparaciones de modelos y procedimientos

Es la parte que más valora el profesor. Hay **dos comparaciones**, cada una diseñada **antes** de ver resultados, reportada tal cual salga y **corta**: una tabla o un gráfico y unas pocas frases de conclusión.

1. **Modelos de sentimiento (notebooks 07 y 08).** La escalera M0–M3b de 8.6 sobre PhraseBank (F1 macro, exactitud, matriz de confusión) y sobre los ~200 titulares etiquetados a mano. Preguntas: ¿cuánto aporta preentrenar (M2 vs M3a)? ¿Encoder o decoder (M3a vs M3b)?
2. **Noticias frente a filings SEC.**
   - **Descriptiva (notebook 08):** distribución del sentimiento de cada fuente y grado de acuerdo entre noticias y filings de la misma empresa en el mismo periodo.
   - **Utilidad (notebook 09):** backtesting de las reglas solas, de las reglas con veto por sentimiento negativo de noticias y de las reglas con veto por sentimiento negativo de filings. Las tres variantes usan las mismas empresas y el mismo periodo, en el que hay datos de ambas fuentes (oct 2016 – jun 2020).

### 8.10 Interfaz (`app.py`, último paso)

Un único archivo Streamlit. Lee los resultados de `data/` y usa `funciones.py`; no recalcula nada por su cuenta.

- **Sidebar:** selector de tickers/sectores y sliders de sensibilidad (umbrales de RSI, ADX, CMF…).
- **Vista general (screener):** tabla de activos con señal del día (riesgo, sentimiento con veto, rango de precio esperado con `calcular_rango_esperado`).
- **Vista de detalle:** velas con Fibonacci, Bollinger, MACD y RSI; stops; panel de estacionalidad; alertas de sentimiento indicando su fuente (noticia o filing).
- Disclaimer visible.

### 8.11 Fuera de alcance — trabajo futuro

Se mencionan en la memoria como **trabajo futuro**, sin implementarse:

- Comparaciones descartadas el 19 oct 2026 para acortar el trabajo: reglas explícitas frente a regresión logística; sentimiento por empresa frente a por sector; señales con régimen frente a sin régimen.
- FinBERT preentrenado como techo de referencia; ajuste del BERT con capas congeladas frente a descongeladas.
- Más histórico de noticias: el fichero de FNSPID con artículos completos hasta 2023 (23 GB), GDELT (2017–hoy, solo titulares y muy limitado por peticiones) y el guardado diario de RSS (Yahoo, Google News, `yfinance`).
- Bloques temáticos de empresas (renovables, automóvil, cripto) por palabras clave del Item 1.
- Momentum cross-sectional, amplitud de mercado (Oscilador McClellan), automatización diaria del screener y notificaciones.

**Si vamos justos se recorta, en este orden:** la parte AllAgree, pulir `app.py`. Nunca el sentimiento ni las dos comparaciones.

---

## 9. Formato de respuesta esperado del agente

En cada paso:

1. Decir qué notebook (o sección) y qué funciones de `funciones.py` se van a crear o modificar, y por qué.
2. Construir solo ese paso.
3. Explicar cómo comprobarlo: *Restart & Run All* y qué resultado debe verse (cifras, gráficos, comprobaciones en verde).
4. Señalar qué queda para el siguiente paso.
5. Esperar confirmación antes de avanzar.

---

## 10. Documentación en paralelo — bitácora

Cada decisión relevante (fuente de datos, cambio de enfoque, umbral elegido, resultado de un experimento, alternativa descartada) se registra **en el momento** en la bitácora del proyecto:

**https://claude.ai/artifact/7URU3PBn8nQMBCRb1qvZCE** — "Bitácora — Screener Financiero EEUU (PCD)", documento de Claude Docs.

- Escribir siempre en **ese mismo documento**; nunca crear una bitácora nueva.
- Formato de cada entrada: **Contexto** (qué lo origina) · **Decisión** (en una frase) · **Razonamiento** (por qué, qué se descartó) · **Alternativas consideradas** · **Estado** (documentado / implementado / pendiente).
