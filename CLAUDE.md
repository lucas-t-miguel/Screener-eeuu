# CLAUDE.md — ¿Mueven las noticias la bolsa? (Proyecto de Ciencia de Datos)

Debe leerse íntegramente antes de generar o modificar código.

> **Última revisión: 19 nov 2026** (revisión final antes de la entrega; ver el último párrafo de este encabezado). Cambios del 1 nov (simplificación para dominar todo el código en la defensa): la **pregunta central** pasa a ser si hay relación entre las noticias y la bolsa; **híbrido**: se mantienen los indicadores y las señales técnicas (04 y 05), pero salen los stops, el Sharpe, la estacionalidad, el backtesting completo y el screener; **tres modelos de sentimiento** (diccionario, Transformer y BERT) con **código como el de los apuntes de clase**; los filings quedan solo como corpus de preentrenamiento. Antes: notebooks y alcance de Trade Recommender (11 oct), modelos propios y plan reducido (19 oct). El razonamiento de cada cambio está en la bitácora (sección 10).
>
> **17 nov 2026:** entrega el **6 dic 2026** (memoria técnica, presentación de ~15 min y el proyecto sin este fichero). Recortes: sin desglose por sector y 09 simplificado (`app.py` salió ese día y volvió el 18 nov como explorador para la presentación, 8.10). Se añade **M4, FinBERT preentrenado** (sugerencia del tutor), sin quitar M1–M3; se añade después de haber mirado el test de M1–M3 (8.6).
>
> **18–19 nov 2026, revisión final:** la app existe (cuatro páginas, 8.10); `05_senales_y_riesgo.ipynb` pasa a `05_senales.ipynb`; el 02 guarda en caché sus consultas de ejemplo a yfinance y el 06 solo pide noticias a NewsAPI con `ACTUALIZAR_NEWSAPI = True`, así que los nueve notebooks corren sin red con la caché; el 08 añade **M4 − M1** con *bootstrap* emparejado por evento (8.8); cada sección de los notebooks empieza con su pregunta y acaba con su conclusión.

---

## 0. Contexto y relación con el proyecto anterior

Este proyecto **empieza de cero** como práctica de la asignatura **Proyecto de Ciencia de Datos**. Existe un proyecto personal previo, **Trade Recommender**, que ya validó gran parte del enfoque técnico (indicadores, régimen de mercado, señales). Es **referencia y fuente de los parámetros**, no código a copiar: aquí todo se reescribe de forma más simple y legible.

**Diferencias clave respecto al proyecto anterior:**

1. **Alcance de mercado:** solo acciones de EEUU (NASDAQ/NYSE).
2. **Nueva fuente de datos: SEC EDGAR (sec.gov)**, con APIs oficiales y descarga/parseo de los informes de las empresas (8.7).
3. **El análisis de sentimiento se completa aquí** (quedó pendiente en Trade Recommender), con **texto no estructurado** de dos orígenes: titulares de noticias (sentimiento) e informes de la SEC (corpus de preentrenamiento).
4. **Formato académico:** notebooks legibles de arriba abajo.
5. **Documentación en paralelo obligatoria** en la bitácora (sección 10).

---

## 1. Identidad y rol del agente

Actúas como **ingeniero de software senior especializado en ciencia de datos en Python**, colaborando con un estudiante de Ciencia de Datos e IA. El proyecto tiene triple propósito:

1. **Académico (prioritario):** un pipeline de datos completo y metodológicamente honesto: ingesta de fuentes heterogéneas, tratamiento de texto no estructurado, comparación de modelos y evaluación crítica de resultados, incluidos los resultados nulos.
2. **Formativo:** el autor debe **entender y poder defender cada línea** ante el tribunal. Ante la duda entre una solución sofisticada y una legible, elige la legible. Si una técnica está en los apuntes de clase, se implementa como en los apuntes (regla 11).
3. **Funcional:** medir de forma reproducible si las noticias se relacionan con el movimiento de las acciones.

---

## 2. Criterios de evaluación de la asignatura

El profesor **no evalúa el rendimiento económico**, sino **cómo se tratan los datos**. Pide explícitamente:

- **Tratamiento de datos no estructurados** (obligatorio → el sentimiento de los titulares y el corpus de informes son la columna vertebral del proyecto).
- **Comparaciones entre modelos o procedimientos** (le gustan especialmente → sección 8.9).

Se evalúa: dificultad, originalidad, desarrollo de la práctica en clase a lo largo del curso, calidad del artículo, resultados obtenidos, preparación y ejecución de la presentación, y adecuación de las respuestas a las preguntas.

**Implicación para el agente:** prioriza el rigor en el tratamiento del texto, comparaciones bien diseñadas y resultados honestos y explicables. Prefiere menos piezas bien entendidas a más piezas difíciles de defender.

---

## 3. Descripción del proyecto

**Pregunta central:** ¿se relaciona el sentimiento de las noticias de una empresa con el movimiento posterior de su acción?

Se responde en tres partes:

1. **El instrumento de medida (notebook 07):** ¿qué forma de entrenar un modelo mide mejor el sentimiento de un texto financiero? Tres modelos, de no entrenar nada a preentrenar y ajustar.
2. **La relación noticias–bolsa (notebook 08):** rentabilidad de la acción en los días siguientes a una noticia positiva, neutral o negativa.
3. **Noticias y análisis técnico (notebook 09):** ¿cambia lo que pasa después de una señal técnica de compra o venta según las noticias previas?

**Sin predicción de precios:** los modelos clasifican el sentimiento de un texto. La relación con la bolsa se mide con rentabilidades observadas y contrastes estadísticos sencillos, nunca con un modelo que prediga precios.

**Honestidad metodológica:** si una hipótesis da un resultado nulo o débil, se documenta tal cual. Un hallazgo negativo bien medido es un resultado válido.

---

## 4. Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12+ (lo exige `pandas-ta==0.4.71b0`; instalado: 3.13) |
| Entorno de trabajo | Jupyter notebooks (`.ipynb`) + un `funciones.py` compartido |
| Datos de mercado | `yfinance` |
| Análisis técnico | `pandas`, `pandas-ta` |
| Visualización | `plotly` |
| Datos SEC | API oficial EDGAR (`data.sec.gov/submissions`) + `requests` y `BeautifulSoup` para el texto de los informes |
| Noticias | FNSPID (titulares 2016–2020) y NewsAPI (plan gratuito, último mes) |
| Sentimiento (PLN) | Diccionario Loughran-McDonald y modelos propios en `keras` 3 sobre TensorFlow, escritos **como en los apuntes de clase**: `TextVectorization` por palabras, capas `TransformerEncoder` y `PositionalEmbedding`, y el enmascarado de `bert_simplificado`. **M4:** FinBERT preentrenado (`yiyanghkust/finbert-pretrain`, Apache-2.0) cargado con **KerasHub** (`keras-hub` 0.33, `tensorflow-text` 2.21.1) sobre TensorFlow; pesos del commit `88ab954`, convertidos una vez a `safetensors` con `torch` en el procesador (solo para esa conversión). Ni Claude ni otro LLM por API |
| Evaluación | `scikit-learn` (partición, F1, matriz de confusión) |
| Interfaz | Streamlit (≥ 1.66): app de cuatro páginas que solo lee `data/cache/` (8.10) |
| Claves | `.env` + `python-dotenv` |

> **Apuntes de clase:** carpeta `/Users/lucas/Desktop/4º carrera/AAPI (Algoit. y arquit. para el procesado de imágenes)/3. NLP`. Referencias principales: `1. transformers/Modelos simples NLP de clasificacion.ipynb`, `1. transformers/Transformers.ipynb` y `2. BERT/bert_simplificado.ipynb`. Están en TensorFlow 2.13 / Keras 2 y clasifican en 2 clases (IMDB). Aquí se adaptan a Keras 3 y a 3 clases, con el menor número de cambios posible.

> **Motor de Keras: TensorFlow** (decisión del autor, 1 nov 2026, tras la prueba descrita en la bitácora). Es el motor de clase y el que pide menos cambios al código de los apuntes: con PyTorch hacían falta 5 cambios más (tensores en la GPU que no pasan a numpy, `tf.*` dentro de las capas, `float64`, el embedding de posición fuera de una capa) y la GPU del Mac cerraba Python en `torch.arange`. Tampoco era más rápido con esta red: 90 ms por lote con TensorFlow frente a 96 ms con PyTorch (preentrenamiento, lotes de 32). Todos los notebooks con Keras lo fijan en su primera celda de código, antes de importarlo: `os.environ["KERAS_BACKEND"] = "tensorflow"`.

> **Cambios a los apuntes por Keras 3** (cada uno se anota en el notebook como "Diferencia con los apuntes"):
> 1. Nombres de capa sin `/` (`encoder_0/ffn` → `encoder_0_ffn`): Keras 3 no los admite.
> 2. Pérdida del preentrenamiento con la reducción por defecto en lugar de `Reduction.NONE`: los `sample_weights` ya anulan las posiciones no tapadas.
> 3. Token de máscara `"[mask]"` en minúscula como valor por defecto de `special_tokens`: en los apuntes el valor por defecto es `"[MASK]"`, que la estandarización convertiría en `[UNK]`, pero la llamada de los apuntes ya pasa `"[mask]"`, así que el resultado es el mismo (no es un error de los apuntes).
> 4. `.keras` en lugar de `.h5`, `pd.concat` en lugar de `DataFrame.append`, y 3 clases (`Dense(3, activation="softmax")` y `sparse_categorical_crossentropy`).
> 5. En `PositionalEmbedding.compute_mask`, `keras.ops.not_equal` en lugar de `tf.math.not_equal`: Keras 3 la llama con un tensor simbólico al construir el modelo, también con TensorFlow (medido el 1 nov).

> **Sin entorno virtual:** por decisión consciente del autor, las dependencias se instalan en el Python global (`python3 -m pip install -r requirements.txt`). Reversible en cualquier momento.

> **Sin Selenium ni Playwright:** los informes de EDGAR son HTML estático; `requests` + `BeautifulSoup` basta.

---

## 5. Reglas operativas de desarrollo (obligatorias)

1. **Modular e iterativo.** Nunca generar el proyecto completo ni un bloque entero en una sola respuesta. Un paso = un notebook, o una sección de un notebook.
2. **De lo básico a lo complejo.** Cada notebook debe funcionar y verificarse antes de pasar al siguiente.
3. **Un paso a la vez.** Al terminar cada entregable, explicar qué se ha construido y cómo comprobarlo, y **esperar confirmación** antes de seguir.
4. **Nunca asumir el formato de datos reales.** Antes de escribir un parser, mirar la respuesta real de la fuente y mostrarla en el notebook.
5. **Pocos notebooks, cada uno con un tema.** Cada sección empieza con la pregunta que responde y termina con una frase de conclusión. Lo que se usa en más de un sitio vive en `funciones.py`, **nunca copiado** entre notebooks.
6. **Sin modelos de caja negra para precios.** Ningún modelo predice precios. Los modelos solo clasifican sentimiento, y la relación con la bolsa se mide con rentabilidades observadas.
7. **Errores y límites de API explícitos.** Toda llamada externa contempla timeouts, límites y datos faltantes, y falla con un mensaje claro. SEC EDGAR: `User-Agent` con contacto real y máximo ~10 peticiones/segundo.
8. **Disclaimer permanente.** Nada de lo que muestre el proyecto es una recomendación de inversión.
9. **Documentar decisión y razonamiento en el momento**, en la bitácora (sección 10).
10. **Legibilidad por encima de todo.** Nombres en español y descriptivos, celdas cortas, nada de abstracciones innecesarias. Si una celda no se explica en 30 segundos, se simplifica o su detalle va a la bitácora.
11. **Como en los apuntes.** Si un modelo o técnica aparece en los apuntes de clase, se usa la misma estructura y los mismos nombres. Cada diferencia se anota en el notebook con una línea **"Diferencia con los apuntes: …"** y su motivo (Keras 3, 3 clases, validación propia…).
    **Excepción: M4** (8.6) parte de un modelo que no está en los apuntes: se carga con KerasHub en lugar de las capas de clase, usa el tokenizador WordPiece de FinBERT en lugar de `TextVectorization`, se ajusta en una sola fase con Adam(2e-5) y clasifica con la salida de `[CLS]`. Cada una de estas diferencias se anota igualmente con su línea "Diferencia con los apuntes".

---

## 6. Estructura del proyecto

```
PCD/
├── .env                          # claves reales (NO se sube a git)
├── .env.example                  # plantilla sin valores
├── .gitignore
├── requirements.txt
├── README.md
├── CLAUDE.md
├── funciones.py                  # funciones reutilizables (datos, indicadores, señales, textos, evaluación)
├── capas_keras.py                # piezas de Keras de M2, M3 y M4 (fija TensorFlow); las importan el 07 y el 08
├── 01_universo_tickers.ipynb     # SEC: lista de empresas → NASDAQ + NYSE
├── 02_precios_y_recorte.ipynb    # yfinance: precios + recorte a 1.000 empresas + 10 años de histórico
├── 03_sec_datos_estructurados.ipynb  # SEC: submissions (informes + sector Fama-French)
├── 04_indicadores_tecnicos.ipynb # los 5 indicadores de las señales: ADX, EMA, CMF, RSI, Donchian
├── 05_senales.ipynb              # régimen de mercado y señales de compra/venta
├── 06_textos.ipynb               # noticias (FNSPID + NewsAPI), informes SEC y corpus de preentrenamiento
├── 07_modelos_sentimiento.ipynb  # PhraseBank; M1 diccionario, M2 Transformer, M3 BERT, M4 FinBERT; comparación
├── 08_noticias_y_bolsa.ipynb     # sentimiento de los titulares, 300 titulares a mano, estudio de eventos y M4 − M1
├── 09_senales_y_noticias.ipynb   # señales técnicas con y sin noticias
├── app.py                        # app de Streamlit (8.10): navegación, CSS mínimo y plantilla de Plotly
├── paginas/                      # una página por fichero (proyecto, modelos, noticias_y_bolsa, explorador) y estilo.py
├── .streamlit/config.toml        # tema de la app (solo modo claro)
├── etiquetas/                    # titulares etiquetados a mano y su regla (en git)
├── resultados/                   # predicciones del primer M3, guardadas antes de su corrección (en git)
├── archivo/                      # versiones retiradas (no se sube a git; el historial está en git)
└── data/                         # no se sube a git
    ├── cache/                    # todo lo descargado o calculado, en parquet (y las respuestas de ejemplo de yfinance, NewsAPI y la SEC)
    ├── raw/fnspid/               # FNSPID original (5,7 GB)
    ├── phrasebank/               # Financial PhraseBank (CC BY-NC-SA 3.0, Malo et al. 2014)
    ├── loughran_mcdonald/        # diccionario Loughran-McDonald (descargado a mano; lo usa M1)
    └── modelos/                  # pesos de los modelos (.keras, .weights.h5), vocabularios y finbert_pretrain/ (FinBERT convertido)
```

Los notebooks van en la raíz junto a `funciones.py` para que `from funciones import ...` funcione sin trucos de rutas.

---

## 7. Convenciones de los notebooks

- **Primera celda (markdown):** título, objetivo en 2-3 frases, **entradas** (archivos de `data/` que lee) y **salidas** (archivos que escribe).
- **Segunda celda (código):** imports, `load_dotenv()` si el notebook usa claves (y el motor de Keras si hace falta). Nada más.
- **Antes de cada bloque de código, una celda markdown** que explique en lenguaje llano qué se hace y por qué.
- **Independencia entre notebooks:** cada notebook lee sus entradas de `data/` y guarda sus salidas en `data/`. Nunca depende de variables en memoria de otro notebook.
- **Caché obligatoria** para todo lo externo o caro (SEC, yfinance, noticias, entrenamientos y predicciones): si el archivo ya existe en `data/cache/` o `data/modelos/`, se lee de disco. Re-ejecutar un notebook no puede volver a entrenar un modelo ni agotar el cupo diario de una API.
- **Semilla fija** (`SEMILLA = 42` en `funciones.py`) para particiones, inicialización de redes y muestreos.
- **Sección final "Comprobaciones":** celdas con `assert` que verifican los resultados.
- **Cierre de un notebook:** solo tras *Restart & Run All* sin errores. Se guarda **con las salidas visibles**.
- **Commit antes de cualquier cambio grande** en un notebook (norma del autor del 20 oct).
- **Sin referencias internas en lo que se entrega:** los notebooks, `funciones.py` y el `README.md` no citan este CLAUDE.md, la bitácora ni herramientas de IA. Si hace falta remitir a un detalle, se escribe "detalle en la memoria". El autor declara el uso de IA en la memoria y en la presentación.
- `print()` y `display()` permitidos; gráficos con `plotly`.

**Reglas de `funciones.py`:**

- Solo definiciones de funciones y constantes; ningún código que se ejecute al importarlo.
- Cada función con type hints y un docstring corto en español.
- Una función entra en `funciones.py` cuando se usa en más de un sitio o es cálculo central de un notebook que conviene leer aparte (descargas, indicadores, señales). Si no, se define en el notebook.
- Ante datos inválidos, `raise` con un mensaje claro.
- Las funciones que ya no usa ningún notebook se quitan (quedan en el historial de git).

---

## 8. Especificaciones funcionales

### 8.1 Universo de tickers (notebooks 01 y 02) — hecho

- **Fuente:** `https://www.sec.gov/files/company_tickers_exchange.json`. No figura en la documentación oficial de la SEC, pero contiene exactamente los mismos CIK que el fichero documentado `company_tickers.json` (8.017 de 8.017, verificado el 11 oct 2026) y es el único que trae la bolsa.
- **Filtro:** NASDAQ + NYSE → 7.659 tickers (4.359 Nasdaq + 3.300 NYSE, sobre 10.422 entradas).
- **Recorte a 1.000 empresas:** solo acciones ordinarias (`quoteType = EQUITY`) con capitalización ≥ 2.000 M USD, ordenadas por **volumen mediano en dólares de 63 sesiones**; una sola clase de acción por empresa (CIK único). Resultado: 640 de NYSE y 360 de Nasdaq.
- **Histórico:** OHLCV diario ajustado por splits y dividendos, del 10 oct 2016 al 9 oct 2026.
- **Caché de las consultas de ejemplo (19 nov):** la descarga de 3 tickers y el `fast_info` de 5 tickers que enseña el 02 se guardan en `data/cache/yfinance_ejemplo_descarga.parquet` y `yfinance_ejemplo_fast_info.json` (reconstruidas a partir de la salida guardada, consulta del 20 oct 2026); el 02 ya no llama a yfinance si existen.

### 8.2 Indicadores técnicos (notebook 04) — hecho

| Indicador | Parámetros | Uso |
|---|---|---|
| RSI | 14 sesiones; umbrales 30 / 70 | Señal en régimen lateral (8.3) |
| CMF (Chaikin Money Flow) | 20 sesiones; umbral ±0,07 | Confirmación por volumen del cruce de EMA |
| ADX | 14 sesiones; umbral 25 | Régimen de mercado |
| Cruce de EMA | EMA(4) vs EMA(8) | Señal en régimen de tendencia |
| Canal de Donchian | 20 sesiones (sin el día de hoy) | Señal de ruptura en cualquier régimen |

Se usa `pandas-ta`; Donchian y CMF se corrigieron o se escribieron a mano en el notebook 04 (detalle en la bitácora). Solo se calculan los indicadores que usa alguna señal: MACD, Bollinger, Fibonacci y ATR se quitaron el 1 nov 2026 (si `app.py` necesitara alguno para un gráfico, se calcularía allí).

### 8.3 Régimen de mercado y señales (notebook 05) — hecho

**Régimen:** **tendencia** si ADX ≥ 25, **lateral** si ADX < 25.

**Tres mecanismos de entrada independientes** (la venta es el espejo de la compra):

1. **Cruce de EMA confirmado** (solo en tendencia): EMA(4) cruza por encima de EMA(8) **y** CMF ≥ 0,07.
2. **Reversión por RSI** (solo en lateral): el RSI estaba por debajo de 30 y hoy vuelve a cruzarlo hacia arriba.
3. **Ruptura de Donchian** (cualquier régimen): el cierre de hoy supera el máximo de las 20 sesiones anteriores.

Hay señal si se cumple (1) **o** (2) **o** (3).

**Condiciones previas:** calentamiento de 60 sesiones por empresa sin régimen ni señales (el ADX de `pandas-ta` sale inflado al principio) y liquidez mínima de 1 M USD/día (mediana de 20 sesiones del volumen en dólares).

### 8.4 y 8.5 Riesgo y estacionalidad — retirados el 1 nov 2026

Stops (2 × ATR y Chandelier Exit), rango de precio esperado, volatilidad con Sharpe y estacionalidad salen del notebook 05: solo servían para el backtesting completo o el screener, que también salen. La versión completa del 05 queda en `archivo/` y en el historial de git; se citan en la memoria como trabajo realizado y retirado para centrar el proyecto.

### 8.6 Sentimiento (notebooks 06, 07 y 08)

**Textos (notebook 06) — hecho**

- **FNSPID** (Dong, Fan y Peng 2024, arXiv 2402.06698; CC BY-NC 4.0): 464.983 titulares de 639 de las 1.000 empresas, de oct 2016 a jun 2020. Solo cuentan para el sentimiento de una empresa los titulares que la mencionan (`menciona_empresa` en `funciones.py`): 633 empresas con alguno.
- **NewsAPI** (gratuito: 100 peticiones/día, último mes): titulares recientes, de los que salen los 300 etiquetados a mano. Se acumularon ejecutando el 06 del 19 oct al 16 nov; desde el 19 nov el 06 solo pide noticias con `ACTUALIZAR_NEWSAPI = True` (por defecto, False: usa `noticias_newsapi.parquet` y no necesita la clave).
- **Informes SEC:** riesgos y MD&A de los 10-K y 10-Q (oct 2016 – jun 2020), sin las extranjeras (20-F/40-F) y con XOM bajo su CIK antiguo. Se usan **solo como corpus de preentrenamiento** de M3.
- **Corpus de preentrenamiento:** 695.136 frases y 17,1 M de palabras (titulares de FNSPID y hasta 40 frases por informe), solo textos hasta el 30 jun 2020 y sin ninguna noticia de NewsAPI.

**Modelos (notebook 07)**

- **Etiquetas:** Financial PhraseBank, variante 50Agree sin repetidas: 4.836 frases. **Partición estratificada 60/20/20** con `SEMILLA = 42`, la misma para todos los modelos (hecho). La validación elige umbrales, tamaños y épocas; **el test se mira una sola vez**, al final, con todos los modelos a la vez. (Diferencia con los apuntes: allí se valida sobre el test, que es una fuga de información.)
- **Métricas:** F1 macro (principal), exactitud y matriz de confusión, con **intervalo de confianza del 95 % del F1 macro por bootstrap** (1.000 remuestreos), en una única función de `funciones.py` que también usa el 08. Referencia mínima: decir siempre "neutral" (F1 macro 0,248 en validación). Sensibilidad: el subconjunto del test que está en AllAgree.
- **Cuatro modelos, cuatro formas de conseguir un modelo de sentimiento** (M4 añadido el 17 nov):

| Modelo | Qué es | Cómo aprende | Apuntes de referencia |
|---|---|---|---|
| M1 | Diccionario Loughran-McDonald: cuenta palabras positivas y negativas | No se entrena; solo el umbral de "neutral" se elige en validación | — (fuera de los apuntes: es el estándar en finanzas) |
| M2 | Transformer encoder con embedding posicional | Solo con las frases de PhraseBank | `Transformers.ipynb` (`TransformerEncoder`, `PositionalEmbedding`) |
| M3 | La misma red que M2, preentrenada como BERT | Primero adivina palabras tapadas en el corpus propio; después se ajusta con PhraseBank como en clase (congelado y después descongelado) | `bert_simplificado.ipynb` |
| M4 | FinBERT: BERT base (110 M de parámetros) preentrenado por otros con 4.900 M de palabras financieras (10-K/10-Q, conferencias de resultados e informes de analistas) | Se ajusta entero con PhraseBank en una sola fase: Adam(2e-5), hasta 136 tokens, cabeza `[CLS]` + Dropout(0,1) + Dense(3), sin pesos de clase, EarlyStopping (paciencia 3) | Cercano a `bert_simplificado`, con las excepciones de la regla 11 |

- **M2 y M3 comparten** vocabulario (`TextVectorization` por palabras con el token `[mask]`, aprendido con el corpus y el entrenamiento de PhraseBank), arquitectura (`PositionalEmbedding` + `TransformerEncoder`) y tamaño (el de `bert_simplificado`: 128 dimensiones, 8 cabezas, 128 neuronas; frases de hasta 64 palabras), para que la única diferencia sea el preentrenamiento. Preentrenamiento: el corpus entero con 2 épocas, como en los apuntes (76,6 min el oficial, mezclado, y 80,6 min el primero, según sus historiales en `data/modelos/`), una sola vez y guardado en `data/modelos/`.
- Antes de descargar el diccionario Loughran-McDonald, mostrar su fuente, formato y licencia (regla 4).
- Con el desequilibrio de clases, se valoran pesos de clase en M2 y M3 viendo la validación.
- **M4, FinBERT preentrenado (17 nov, sugerencia del tutor):** lo habitual es partir de un BERT preentrenado a gran escala, porque preentrenar uno propio da peores resultados (M3: 0,586 en el test). Es **solo el modelo preentrenado** (FinVocab en minúsculas, 30.873 piezas) y **no ha visto PhraseBank**. Quedan **excluidos** ProsusAI/finbert (se ajustó con Financial PhraseBank y habría visto el test) y finbert-tone (ya es un clasificador de sentimiento ajustado por otros). **Plan B**, solo si FinBERT no se pudiera cargar (decidido antes de ver resultados): `nlpaueb/sec-bert-base`. **Transparencia:** M4 se añade **después de haber mirado el test de M1–M3**, por sugerencia del tutor y no por los resultados; consta en el notebook y en la bitácora, y todas sus decisiones (136 tokens, Adam(2e-5) fijo, una fase, cabeza `[CLS]`, GELU exacta) se fijan antes de entrenarlo. Detalles técnicos: KerasHub no lee el `config.json` de 2020 (sin `model_type`) ni `pytorch_model.bin`, así que el 07 descarga el commit `88ab954`, comprueba su SHA-256, añade `model_type` y convierte los pesos del encoder a `safetensors` (con caché); KerasHub usa la GELU aproximada y el 07 la cambia por la exacta del BERT original (diferencia máxima con `transformers` de 2,4·10⁻⁵ en la salida del encoder con 64 frases de la validación, medida aparte el 19 nov; con la GELU aproximada, 5·10⁻²). Los pesos ajustados se guardan como `m4_finbert.weights.h5` (1,3 GB: Keras guarda también el estado de Adam). **Resultado (17 nov):** 4 épocas (23,7 min), la mejor la 1; test 0,813 (0,78–0,84); M4 − M2 = +0,140 (0,10–0,18); M4 − M3 = +0,226 (0,19–0,27); acierta 102 de 121 negativas. Mejor instrumento para el 08.
- **Estructura del 07 (recorte del 17–18 nov, sin cambiar ningún resultado):** 1 PhraseBank · 2 partición, métricas y referencia · 3 M1 · 4 vocabulario · 5 M2 · 6 M3 preentrenamiento (ya con la permutación completa) · 7 M3 ajuste y, en 7.4, el fallo del primer preentrenamiento · 8 M4 · 9 examen final de los cuatro con el test (con la cronología: M1–M3 el 16 nov, después la corrección de M3 y M4) · 5 celdas de Comprobaciones. Las tablas usan `tabla_f1` y `tabla_diferencias` de `funciones.py`.

**Aplicación (notebook 08)**

- Los modelos puntúan todos los titulares de FNSPID que mencionan a su empresa (hecho con los cuatro: 192.069 titulares, `data/cache/sentimiento_titulares.parquet`, con caché por modelo; M4, por grupos de longitud de 32, 64 y 136 tokens, porque con la cabeza `[CLS]` el resultado no depende del relleno). Salida: una tabla (parquet) con una fila por titular y modelo: empresa, fecha, titular, sentimiento (−1/0/1) y confianza (probabilidad de la clase elegida; en M1, la proporción de palabras con tono).
- **Validación fuera de dominio:** 300 titulares recientes de NewsAPI (como mucho 5 por empresa; 200 previstos, ampliados a 300 por el autor el 16 nov antes de etiquetar) etiquetados a mano por el autor en `etiquetas/titulares_a_mano.csv` (en git), con la regla de `etiquetas/regla_de_etiquetado.md` y una columna `trata_de_la_empresa` para los que el filtro dejó pasar por error. Se evalúan con la misma función y `comparar_f1` que PhraseBank, con los 300 y solo con los 271 que tratan de la empresa. **Resultado (17 nov):** F1 macro con los 300: M1 0,460, M2 0,376, M3 0,375, **M4 0,626** (0,56–0,69); M4 − M2 = +0,25 (0,18–0,32); M2 − M1 = −0,085 (−0,17 a 0,004), no concluyente. M2 y M3 se hunden fuera de dominio; M4 es el instrumento para el estudio de eventos.
- **`capas_keras.py`, compartido por el 07 y el 08 (decisión del autor, 18 nov; sustituye a la excepción a la regla 5 del 16 nov):** `custom_standardization`, `MAX_LEN`, `VOCAB_SIZE`, las capas `TransformerEncoder` y `PositionalEmbedding` (código de los apuntes) y, para M4, `construir_m4` (sin compilar), `cargar_m4` y el tokenizador. Va fuera de `funciones.py` porque necesita Keras: fija TensorFlow al importarse, así los notebooks 01–06 no cargan Keras. El 08 comprueba que reproduce las predicciones de validación del 07 (100 %).
- **M3 oficial: el corregido** (`m3_mezclado_descongelado.keras`; el 07 lo usa directamente en las secciones 6 y 7, y el fallo se cuenta en la 7.4 con las predicciones del primer M3 guardadas en `resultados/m3_original_predicciones.csv`): el preentrenamiento original mezclaba el corpus con `shuffle(1000)` y dejaba titulares e informes en bloques. Test: 0,586 (0,54–0,62); M3 − M2 = −0,087 (−0,13 a −0,05).

### 8.7 Ingesta desde SEC EDGAR (notebooks 01, 03 y 06) — hecho

1. **Datos estructurados (API oficial):** `data.sec.gov/submissions/CIK##########.json` (lista de informes y código SIC → 12 sectores de Fama-French). Las cifras financieras de 2025 de la API `frames` se retiraron el 18 nov (8.11).
2. **Texto narrativo:** descarga del documento principal con `requests` y extracción de riesgos y MD&A con `BeautifulSoup` y reglas sobre las cabeceras "Item".
3. **Política de acceso:** sin cuenta ni API key. `User-Agent` con contacto real (`SEC_EDGAR_USER_AGENT` del `.env`) y máximo ~10 peticiones/segundo.

### 8.8 Noticias y bolsa (notebooks 08 y 09)

**Medida común:** rentabilidad de la acción **menos la del mercado (SPY)** en un plazo fijo después del evento. SPY se descarga en el 08 (con caché), con el mismo ajuste que el resto de precios.

**Notebook 08 — ¿se mueve la acción después de una noticia?** (estudio de eventos)

- Titulares de FNSPID que mencionan a su empresa, oct 2016 – jun 2020.
- Si una empresa tiene varios titulares el mismo día, se agregan en un sentimiento diario (la regla se decide viendo los datos).
- **Sin mirar al futuro:** FNSPID solo trae el día de publicación (el 96 % de los titulares tiene la hora `00:00:00`, comprobado el 16 nov). La rentabilidad "posterior" es **1 sesión**: del cierre del día de la noticia al cierre siguiente (decisión del autor, 16 nov; se descartó 5 sesiones). Las noticias de fin de semana, desde el último cierre anterior. Varios titulares de una empresa el mismo día: media de sus sentimientos (−1/0/1) y su signo. Intervalo de la diferencia: Welch, con su limitación anotada.
- **Hecho (18 nov):** un evento = empresa y sesión de referencia (las noticias de viernes a domingo forman un solo evento); colas *winsorizadas* a los percentiles 1 y 99 de todas las anormales del periodo (−6,1 % y +6,6 %), con la versión sin recortar como sensibilidad; `diferencia_medias_welch` en `funciones.py`. Resultado, positivos − negativos en la sesión siguiente: M4 +0,21 % (0,17–0,26), M1 +0,14 % (0,10–0,19), M2 y M3 sin diferencia. El mismo día, M4 +1,33 % (unas 6 veces más; relación inflada).
- **M4 − M1 (19 nov, sección 6.4 del 08; pregunta 2 de 8.9):** diferencia de sus «positivos − negativos» en la sesión siguiente con *bootstrap* emparejado por evento (los mismos eventos para los dos modelos, 1.000 remuestreos, SEMILLA): **+0,07 puntos (0,02–0,12)**, guardado en `resultados_08_m4_menos_m1.parquet`. M4 encuentra algo más de relación que el diccionario. Se añadió después de ver la tabla de la 6.2 (consta en el notebook). **Limitación de independencia:** Welch y el *bootstrap* suponen eventos independientes, pero los del mismo día se mueven en parte juntos, así que los intervalos pueden ser algo estrechos, sobre todo el de M4 − M1 (límite inferior cerca de 0); remuestrear por días queda como trabajo futuro.
- La rentabilidad **del mismo día** se muestra aparte y se explica como relación inflada: muchos titulares cuentan un movimiento que ya ha ocurrido ("X shares fall after earnings").
- Resultado: rentabilidad media por grupo (positivo, neutral, negativo), diferencia entre positivo y negativo con intervalo de confianza del 95 %, **para cada uno de los cuatro modelos**. El desglose por sector queda fuera (recorte del 17 nov; se puede añadir si sobra tiempo).

**Notebook 09 — ¿cambian las señales técnicas según las noticias?**

- Señales de compra y de venta del 05, contando solo el **primer día de cada racha** (la ruptura de Donchian se repite varios días seguidos).
- Rentabilidad a 5 y 10 sesiones menos SPY (en las ventas, con el signo cambiado: aciertan si la acción baja).
- Tres grupos: con noticias positivas en los días previos, sin noticias y con noticias negativas. **Diseño simplificado (17 nov), fijado antes de ver los datos:** una sola ventana de noticias previas y un solo modelo de sentimiento, M4 (el mejor en PhraseBank y en los titulares a mano), sin elegir umbrales en 2016–2018. Se puede ampliar si sobra tiempo.
- Solo empresas y fechas con cobertura de noticias (633 empresas, oct 2016 – jun 2020). **Ventana (decisión del autor, 18 nov): 5 sesiones previas a la señal, sin la de la señal; 3 como sensibilidad.** Los eventos de noticias los guarda el 08 en `data/cache/eventos_noticias.parquet`. **Paso 1 hecho (18 nov):** 54.001 señales (primer día de racha) en 621 empresas; no se excluye ninguna empresa: las 7 del precio congelado eran normales en 2016–2020 (lo estaban en 2025–2026) y las 9 que casi no se movían entonces no tienen noticias. **Paso 2 hecho (18 nov):** con la fila de referencia «todas (solo análisis técnico)». Compras positivas − negativas: +0,31 % a 5 sesiones (0,10–0,51) y +0,49 % a 10 (0,23–0,76), por el mal resultado de las compras tras noticias negativas (−0,45 % y −0,64 %); positivas − sin noticias, no concluyente; ventas, nada concluyente. Las señales solas no baten al mercado. `media_con_intervalo` pasa a `funciones.py` (la usan el 08 y el 09). **Sensibilidad de cobertura (19 nov):** 2.958 señales (5,5 %) caen antes de la primera noticia de su empresa o más de 30 días después de la última y cuentan como «sin noticias»; sin ellas, positivas − negativas no cambia (+0,307 y +0,494) y positivas − sin noticias sigue sin ser concluyente. Mismo límite de independencia que el 08.

**Limitaciones que deben documentarse junto a los resultados:**

- **Sesgo de selección:** las 1.000 empresas se eligieron con su tamaño y liquidez de 2026 y se estudian desde 2016. Infla los niveles de rentabilidad, pero afecta mucho menos a las comparaciones entre grupos y entre modelos sobre las mismas empresas.
- Noticias solo de 633 empresas y hasta junio de 2020; solo titulares, sin el texto del artículo.
- Sin costes de operación: no se simula una estrategia, se miden rentabilidades.

### 8.9 Comparaciones de modelos y procedimientos

Es la parte que más valora el profesor. Dos comparaciones, diseñadas **antes** de ver resultados, reportadas tal cual salgan y **cortas** (una tabla o un gráfico y unas pocas frases):

1. **Cuatro formas de conseguir un modelo de sentimiento (07 y 08).** Tabla de M1, M2, M3 y M4 sobre el test de PhraseBank y sobre los 300 titulares a mano. Preguntas: ¿aprender de ejemplos supera a una lista de palabras hecha por expertos (M2 − M1)? ¿Aporta preentrenar (M3 − M2)? Esta es **la única comparación limpia del preentrenamiento**: M2 y M3 comparten vocabulario, arquitectura, tamaño y datos de ajuste. **M4 − M2 y M4 − M3 comparan recetas completas** (partir de un BERT preentrenado por otros frente a entrenar desde cero o preentrenar uno propio), **no el tamaño del preentrenamiento**: entre M3 y M4 cambian a la vez el tamaño de la red, el corpus, el tokenizador, la cabeza y el procedimiento de ajuste. Una diferencia dentro del intervalo de confianza no se presenta como mejora.
2. **Noticias y bolsa (08 y 09).** ¿Hay relación entre el sentimiento y la rentabilidad posterior? ¿El modelo que mejor clasifica encuentra más relación que el diccionario? ¿Cambia el resultado de las señales técnicas con noticias positivas o negativas?

### 8.10 Interfaz (`app.py`) — hecho: app de cuatro páginas para la presentación (decisión del autor, 18 nov)

*Primer diseño (18 nov), sustituido por el rediseño del párrafo siguiente:* se recupera una `app.py` mínima, con un límite de **un día de trabajo**: un explorador para enseñar los datos en la presentación, sin añadir ningún indicador nuevo al análisis. Un solo fichero con Streamlit, legible y de 200–250 líneas como mucho. No importa Keras ni TensorFlow: solo lee parquet de `data/cache/` (precios, indicadores del 04, señales del 05 y el sentimiento de M4 de los titulares) y usa `funciones.py` si hace falta. Barra lateral: empresa (ticker y nombre) y rango de fechas dentro de oct 2016 – jun 2020. Un gráfico plotly con tres paneles sobre el mismo eje de fechas: (1) velas, EMA(4) y EMA(8), canal de Donchian, señales de compra y venta y días con noticias coloreados según M4 (titular al pasar el ratón); (2) RSI con 30 y 70; (3) ADX con 25. Debajo, la tabla de titulares del periodo con su sentimiento de M4, y el disclaimer visible. Inspiración de diseño (sin copiar código): la vista de detalle de Trade Recommender; no se usa nada de lo retirado (MACD, Bollinger, Fibonacci, amplitud, estacionalidad) ni FastAPI, JavaScript o CSS propio.

**Rediseño (decisión del autor, 18 nov; límite: un día de trabajo):** navegación con `st.navigation` y `st.Page`, y cuatro páginas en `paginas/`: *Proyecto* (la pregunta, los datos, el método en 4 pasos y el aviso, en una pantalla), *Modelos de sentimiento* (F1 macro con su intervalo de los 4 modelos en el test de PhraseBank y con los 300 titulares a mano), *Noticias y bolsa* (positivos − negativos por modelo en la sesión siguiente, el mismo día aparte como relación inflada, y las compras del 09) y *Explorador* (la vista actual, con la empresa y las fechas arriba de la página). **La app no recalcula resultados:** el 07, el 08 y el 09 guardan sus tablas finales en `data/cache/resultados_*.parquet` y la app solo las lee (también `resultados_08_m4_menos_m1.parquet`, en una línea bajo el gráfico de *Noticias y bolsa*; el Explorador lee además los datos diarios del 02, 04, 05 y 08 y `empresas_sec.parquet` del 03). Estilo con la paleta de Trade Recommender, solo modo claro (fondo #f7f7f5, superficie #ffffff, texto #1c1c1e, secundario #6e6e73, acento #0066cc, compra/positivo #1c8a4b, venta/negativo #c1362d, neutral #8e8e93, bordes rgba(0,0,0,0.08), radio 10 px, fuente del sistema): tema en `.streamlit/config.toml`, una plantilla de Plotly común y CSS mínimo en un solo sitio, sin depender de clases internas de Streamlit. Sin Keras ni TensorFlow.

### 8.11 Fuera de alcance — trabajo futuro

Se mencionan en la memoria, sin implementarse:

- **Retirado el 18 nov 2026:** las cifras financieras de 2025 (ingresos, beneficio, activos, patrimonio, margen neto y ROE) de la API `frames` de la SEC, que eran la sección 7 del notebook 03: no las usaba ninguna otra parte del proyecto (eran contexto para la primera `app.py`, que salió el 17 nov y volvió el 18 nov sin ellas).
- **Retirado el 1 nov 2026:** los indicadores que no usa ninguna señal (MACD, Bollinger, Fibonacci y ATR); backtesting completo (stops, tamaño de posición, cartera, CAGR, drawdown) y screener del día; rango de precio esperado, volatilidad con Sharpe y estacionalidad; sentimiento de los informes SEC y la comparación noticias frente a informes; TF-IDF con red densa; GPT propio (encoder frente a decoder, perplejidad); tokenizador WordPiece.
- **Retirado el 19 oct 2026:** reglas explícitas frente a regresión logística; sentimiento por empresa frente a por sector; señales con régimen frente a sin régimen; FinBERT de ProsusAI y finbert-tone (ya ajustados con datos de sentimiento; el de ProsusAI, con PhraseBank, así que evaluarlo ahí no sería limpio; el FinBERT **solo preentrenado** entra como M4 el 17 nov); más histórico de noticias (FNSPID de 23 GB, GDELT, RSS diario); bloques temáticos; 8-K.
- Momentum cross-sectional, amplitud de mercado, automatización diaria y notificaciones.

**Si vamos justos se recorta, en este orden:** el desglose por sector (ya recortado), la parte AllAgree, la app. Nunca el sentimiento ni las dos comparaciones.

- **Trabajo futuro anotado en la revisión final (19 nov):** remuestrear por días (no por eventos) los intervalos del 08 y el 09, para tener en cuenta que los eventos del mismo día se mueven juntos.

---

## 9. Formato de respuesta esperado del agente

En cada paso:

1. Decir qué notebook (o sección) y qué funciones de `funciones.py` se van a crear o modificar, y por qué.
2. Construir solo ese paso.
3. Explicar cómo comprobarlo: *Restart & Run All* y qué resultado debe verse.
4. En cada decisión pendiente, dar 2 o 3 opciones con pros y contras y una recomendación.
5. Señalar qué queda para el siguiente paso y esperar confirmación.

---

## 10. Documentación en paralelo — bitácora

Cada decisión relevante se registra **en el momento** en la bitácora del proyecto:

**https://claude.ai/artifact/7URU3PBn8nQMBCRb1qvZCE** — "Bitácora — Screener Financiero EEUU (PCD)", documento de Claude Docs.

- Escribir siempre en **ese mismo documento**; nunca crear una bitácora nueva.
- Formato de cada entrada: **Contexto** · **Decisión** (en una frase) · **Razonamiento** (por qué, qué se descartó) · **Alternativas consideradas** · **Estado** (documentado / implementado / pendiente).
- Mantener al día la sección "Próximos pasos" al final del documento.
