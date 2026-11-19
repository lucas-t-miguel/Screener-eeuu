"""Funciones reutilizables del Screener Financiero EEUU.

Aquí solo hay definiciones de funciones y constantes: importar este
archivo no ejecuta nada. Cada función está aquí porque se usa en más de
un notebook (o en un notebook y en app.py).
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# --- Rutas ---
CARPETA_CACHE = Path(__file__).resolve().parent / "data" / "cache"

# --- Ficheros descargados a mano (notebooks 03, 06 y 07) ---


def comprobar_descarga_manual(ruta: Path, sha256_esperado: str, enlace_descarga: str) -> None:
    """Comprueba que un fichero descargado a mano está en su sitio y es exactamente el esperado.

    Recibe la ruta donde debe estar, su huella SHA-256 y el enlace de donde se descarga.
    La huella se calcula leyendo el fichero por bloques, sin cargarlo entero en memoria.
    Lanza FileNotFoundError si no está (diciendo de dónde bajarlo) y RuntimeError si la
    huella no coincide (otra versión del fichero o una descarga incompleta).
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}: descárgalo de {enlace_descarga} y guárdalo como {ruta}")
    with open(ruta, "rb") as fichero:
        huella = hashlib.file_digest(fichero, "sha256").hexdigest()
    if huella != sha256_esperado:
        raise RuntimeError(f"{ruta.name} no es el fichero esperado (su huella SHA-256 no coincide): "
                           f"¿otra versión o una descarga incompleta? Vuelve a bajarlo de {enlace_descarga}")


# --- Reproducibilidad ---
# Semilla de todas las particiones, muestreos e inicializaciones de redes, para que Restart & Run All dé las mismas cifras
SEMILLA = 42

# --- SEC EDGAR ---
URL_TICKERS_CON_BOLSA = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TIMEOUT_SEGUNDOS = 10
# La SEC permite como máximo ~10 peticiones por segundo: esperamos 0,1 s tras cada una.
SEC_PAUSA_ENTRE_PETICIONES = 0.1
BOLSAS_DEL_PROYECTO = ("Nasdaq", "NYSE")


def obtener_user_agent_sec() -> str:
    """Devuelve el User-Agent para SEC EDGAR leído de la variable SEC_EDGAR_USER_AGENT.

    Requiere haber llamado antes a load_dotenv() en el notebook.
    Lanza RuntimeError si la variable no está definida, porque la SEC
    rechaza peticiones sin un User-Agent con datos de contacto reales.
    """
    user_agent = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT no está configurado en .env. La SEC exige un "
            "User-Agent con contacto real, p. ej. 'Nombre Apellido email@dominio.com'."
        )
    return user_agent


def descargar_json_sec(url: str, ruta_cache: Path | None = None) -> dict:
    """Descarga un JSON de SEC EDGAR, con caché opcional en disco.

    Recibe la URL y, opcionalmente, la ruta de un archivo .json de caché:
    si ese archivo ya existe se lee de disco y no se hace ninguna petición.
    Devuelve el JSON decodificado tal cual lo sirve la SEC.
    Lanza RuntimeError si hay un fallo de red, timeout o un código HTTP de error.
    """
    if ruta_cache is not None and ruta_cache.exists():
        return json.loads(ruta_cache.read_text(encoding="utf-8"))

    cabeceras = {"User-Agent": obtener_user_agent_sec()}
    try:
        respuesta = requests.get(url, headers=cabeceras, timeout=SEC_TIMEOUT_SEGUNDOS)
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Fallo de red al descargar {url}: {error}") from error
    finally:
        time.sleep(SEC_PAUSA_ENTRE_PETICIONES)

    if respuesta.status_code != 200:
        raise RuntimeError(f"SEC EDGAR devolvió HTTP {respuesta.status_code} al pedir {url}")

    datos = respuesta.json()
    if ruta_cache is not None:
        ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        ruta_cache.write_text(json.dumps(datos), encoding="utf-8")
    return datos


def cargar_universo_tickers(forzar_descarga: bool = False) -> pd.DataFrame:
    """Devuelve el universo completo de tickers de la SEC (todas las bolsas, sin filtrar).

    Usa la caché data/cache/company_tickers_exchange.parquet si existe;
    con forzar_descarga=True la ignora y vuelve a descargar de la SEC.
    Devuelve un DataFrame con columnas cik (texto de 10 dígitos),
    ticker (mayúsculas), name y exchange (puede ser None).
    """
    ruta_parquet = CARPETA_CACHE / "company_tickers_exchange.parquet"
    if ruta_parquet.exists() and not forzar_descarga:
        return pd.read_parquet(ruta_parquet)

    # La SEC sirve un formato tabular: {"fields": [...], "data": [[fila], [fila], ...]}
    crudo = descargar_json_sec(URL_TICKERS_CON_BOLSA)
    universo = pd.DataFrame(crudo["data"], columns=crudo["fields"])
    universo["cik"] = universo["cik"].astype(str).str.zfill(10)
    universo["ticker"] = universo["ticker"].astype(str).str.upper()
    universo = universo[["cik", "ticker", "name", "exchange"]]

    ruta_parquet.parent.mkdir(parents=True, exist_ok=True)
    universo.to_parquet(ruta_parquet, index=False)
    return universo


def filtrar_por_bolsa(
    universo: pd.DataFrame, bolsas: tuple[str, ...] = BOLSAS_DEL_PROYECTO
) -> pd.DataFrame:
    """Se queda solo con los tickers que cotizan en las bolsas indicadas (por defecto NASDAQ y NYSE).

    Recibe el DataFrame de cargar_universo_tickers y devuelve una copia
    filtrada con el índice renumerado desde 0.
    """
    return universo[universo["exchange"].isin(bolsas)].reset_index(drop=True)


def obtener_cik(ticker: str, universo: pd.DataFrame) -> str:
    """Devuelve el CIK (texto de 10 dígitos) de un ticker, sin distinguir mayúsculas.

    Recibe el ticker y el DataFrame del universo donde buscarlo.
    Lanza ValueError si el ticker no está en ese universo.
    """
    coincidencias = universo.loc[universo["ticker"] == ticker.upper(), "cik"]
    if coincidencias.empty:
        raise ValueError(f"El ticker '{ticker}' no está en el universo de la SEC.")
    return coincidencias.iloc[0]


# --- Indicadores técnicos ---
RSI_SESIONES = 14
CMF_SESIONES = 20
ADX_SESIONES = 14
EMA_RAPIDA, EMA_LENTA = 4, 8
DONCHIAN_SESIONES = 20


def calcular_cmf(alto: pd.Series, bajo: pd.Series, cierre: pd.Series, volumen: pd.Series, sesiones: int) -> pd.Series:
    """Chaikin Money Flow: suma de (multiplicador × volumen) / suma de volumen en N sesiones.

    El multiplicador ((cierre - mínimo) - (máximo - cierre)) / (máximo - mínimo) vale 1 si se
    cierra en el máximo y -1 si se cierra en el mínimo. Si máximo = mínimo se toma 0 (neutro).
    Devuelve NaN cuando no hubo ningún volumen en la ventana (no hay flujo que medir).
    """
    rango = alto - bajo
    multiplicador = ((cierre - bajo) - (alto - cierre)) / rango.where(rango > 0)
    multiplicador = multiplicador.fillna(0)
    volumen_ventana = volumen.rolling(sesiones).sum()
    return (multiplicador * volumen).rolling(sesiones).sum() / volumen_ventana.where(volumen_ventana > 0)


def calcular_indicadores(precios: pd.DataFrame) -> pd.DataFrame:
    """Calcula los indicadores técnicos que usan las señales (RSI, CMF, ADX, EMA y Donchian) para UN ticker.

    Recibe un DataFrame con columnas open, high, low, close y volume,
    ordenado por fecha (una fila por sesión). Devuelve un DataFrame con el
    mismo índice y una columna por indicador. Los primeros días de cada
    indicador quedan vacíos (NaN) hasta que hay sesiones suficientes.
    Lanza ValueError si faltan columnas o las fechas no están ordenadas.
    """
    columnas_necesarias = {"open", "high", "low", "close", "volume"}
    if not columnas_necesarias <= set(precios.columns):
        raise ValueError(f"Faltan columnas de precios: {columnas_necesarias - set(precios.columns)}")
    if not precios.index.is_monotonic_increasing:
        raise ValueError("Los precios deben estar ordenados por fecha")

    alto, bajo, cierre, volumen = precios["high"], precios["low"], precios["close"], precios["volume"]
    indicadores = pd.DataFrame(index=precios.index)

    # RSI: fuerza relativa de subidas frente a bajadas (0-100)
    indicadores["rsi"] = ta.rsi(cierre, length=RSI_SESIONES)

    # CMF: presión compradora (+) o vendedora (-) ponderada por volumen, entre -1 y 1.
    # Calculado a mano: pandas-ta divide por un número diminuto cuando máximo = mínimo
    # (días casi sin negociación) y da valores absurdos; aquí ese día cuenta como neutro (0).
    indicadores["cmf"] = calcular_cmf(alto, bajo, cierre, volumen, CMF_SESIONES)

    # ADX: fuerza de la tendencia (sin dirección), 0-100
    indicadores["adx"] = ta.adx(alto, bajo, cierre, length=ADX_SESIONES).iloc[:, 0]

    # Medias exponenciales rápida y lenta para el cruce
    indicadores["ema_rapida"] = ta.ema(cierre, length=EMA_RAPIDA)
    indicadores["ema_lenta"] = ta.ema(cierre, length=EMA_LENTA)

    # Canal de Donchian de las 20 sesiones ANTERIORES (shift(1) excluye hoy): la ruptura
    # compara el cierre de hoy con este canal; si incluyera hoy, el cierre nunca lo superaría
    donchian = ta.donchian(alto, bajo, lower_length=DONCHIAN_SESIONES, upper_length=DONCHIAN_SESIONES)
    indicadores["donchian_minimo_previo"] = donchian.iloc[:, 0].shift(1)
    indicadores["donchian_maximo_previo"] = donchian.iloc[:, 2].shift(1)

    return indicadores


# --- Régimen de mercado y señales ---
ADX_UMBRAL_TENDENCIA = 25
RSI_SOBREVENTA, RSI_SOBRECOMPRA = 30, 70
CMF_UMBRAL = 0.07
# Filtro de liquidez: sin señal si la mediana del volumen en dólares de las últimas
# 20 sesiones (hasta hoy incluido) no llega a 1 M USD/día. Elegido en el notebook 05 (11 oct 2026).
VOLUMEN_MINIMO_SESIONES = 20
VOLUMEN_MINIMO_DOLARES = 1_000_000
# Calentamiento: los primeros valores del ADX de pandas-ta salen inflados (media ~48 en las
# 30 primeras sesiones frente a ~24 después) y no se estabilizan hasta ~60 sesiones.
# Medido en el notebook 05. Durante el calentamiento no hay régimen ni señales.
SESIONES_CALENTAMIENTO = 60


def clasificar_regimen(adx: pd.Series) -> pd.Series:
    """Clasifica cada día como 'tendencia' (ADX >= 25) o 'lateral' (ADX < 25).

    Recibe la serie del ADX de UN ticker, ordenada por fecha, y devuelve una serie
    de texto con el mismo índice. Las primeras SESIONES_CALENTAMIENTO sesiones y
    los días sin ADX quedan vacíos (None).
    """
    adx_fiable = adx.where(pd.Series(range(len(adx)), index=adx.index) >= SESIONES_CALENTAMIENTO)
    regimen = pd.Series(None, index=adx.index, dtype="object")
    regimen[adx_fiable >= ADX_UMBRAL_TENDENCIA] = "tendencia"
    regimen[adx_fiable < ADX_UMBRAL_TENDENCIA] = "lateral"
    return regimen


def detectar_senales(precios: pd.DataFrame, indicadores: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de compra y venta del proyecto a UN ticker.

    Recibe los precios (close, volume) y los indicadores de calcular_indicadores,
    con el mismo índice de fechas ordenado. Devuelve un DataFrame con el régimen,
    si el día es líquido, una columna booleana por mecanismo y sentido, y la
    columna 'senal' ('compra', 'venta' o None). Un día con compra y venta a la
    vez se marca como 'conflicto' y no genera operación. Durante las primeras
    SESIONES_CALENTAMIENTO sesiones no hay régimen ni ningún mecanismo activo.
    Lanza ValueError si los índices no coinciden.
    """
    if not precios.index.equals(indicadores.index):
        raise ValueError("Precios e indicadores deben tener exactamente las mismas fechas")

    cierre = precios["close"]
    rsi, adx, cmf = indicadores["rsi"], indicadores["adx"], indicadores["cmf"]
    rapida, lenta = indicadores["ema_rapida"], indicadores["ema_lenta"]
    senales = pd.DataFrame(index=precios.index)
    senales["regimen"] = clasificar_regimen(adx)
    en_tendencia = senales["regimen"] == "tendencia"
    en_lateral = senales["regimen"] == "lateral"
    pasado_calentamiento = senales["regimen"].notna()
    volumen_dolares = (cierre * precios["volume"]).rolling(VOLUMEN_MINIMO_SESIONES).median()
    senales["liquido"] = volumen_dolares >= VOLUMEN_MINIMO_DOLARES

    # 1. Cruce de EMA confirmado (solo en tendencia), con la CMF a favor
    senales["compra_ema"] = (rapida.shift(1) <= lenta.shift(1)) & (rapida > lenta) & en_tendencia & (cmf >= CMF_UMBRAL)
    senales["venta_ema"] = (rapida.shift(1) >= lenta.shift(1)) & (rapida < lenta) & en_tendencia & (cmf <= -CMF_UMBRAL)

    # 2. Reversión por RSI (solo en lateral): sale de sobreventa / sobrecompra
    senales["compra_rsi"] = (rsi.shift(1) < RSI_SOBREVENTA) & (rsi >= RSI_SOBREVENTA) & en_lateral
    senales["venta_rsi"] = (rsi.shift(1) > RSI_SOBRECOMPRA) & (rsi <= RSI_SOBRECOMPRA) & en_lateral

    # 3. Ruptura de Donchian (cualquier régimen): cierre fuera del canal de las 20 sesiones anteriores
    senales["compra_donchian"] = (cierre > indicadores["donchian_maximo_previo"]) & pasado_calentamiento
    senales["venta_donchian"] = (cierre < indicadores["donchian_minimo_previo"]) & pasado_calentamiento

    # Combinación: cualquiera de los tres mecanismos vale, solo en días líquidos
    hay_compra = (senales["compra_ema"] | senales["compra_rsi"] | senales["compra_donchian"]) & senales["liquido"]
    hay_venta = (senales["venta_ema"] | senales["venta_rsi"] | senales["venta_donchian"]) & senales["liquido"]
    senales["senal"] = None
    senales.loc[hay_compra, "senal"] = "compra"
    senales.loc[hay_venta, "senal"] = "venta"
    senales.loc[hay_compra & hay_venta, "senal"] = "conflicto"
    return senales


# --- Noticias: a qué empresa se refiere un titular (notebook 06 y app.py) ---
# Si la primera palabra del nombre es una de estas, se usan las dos primeras (TEXAS -> TEXAS INSTRUMENTS)
PALABRAS_DEMASIADO_COMUNES = {
    "AMERICAN", "GENERAL", "FIRST", "UNITED", "NATIONAL", "INTERNATIONAL", "TEXAS", "WESTERN", "EASTERN",
    "SOUTHERN", "NORTHERN", "GLOBAL", "THE", "NEW", "CENTRAL", "PACIFIC", "ATLANTIC", "ADVANCED", "APPLIED",
    "DIGITAL", "PUBLIC", "ROYAL", "STANDARD", "UNIVERSAL", "CAPITAL", "BANK", "ENERGY", "HEALTH", "CHINA",
    "CANADIAN", "PACKAGING", "ALLIED", "CONSOLIDATED", "WASTE",
}

# Palabras clave revisadas a mano (22 oct 2026): las que comparten dos empresas del universo, las que aparecen
# en más titulares (FNSPID + NewsAPI) y las que dieron falsos positivos en NewsAPI. "" = solo vale el ticker.
PALABRA_CLAVE_A_MANO = {
    # Palabra común o de otro significado en titulares financieros
    "TROW": "T ROWE", "TGT": "", "MS": "MORGAN STANLEY", "SWK": "STANLEY BLACK", "TMUS": "T MOBILE",
    "NWSA": "NEWS CORP", "MAA": "MID AMERICA APARTMENT", "TTD": "TRADE DESK", "DOW": "DOW CHEMICAL", "WFC": "WELLS FARGO",
    "GFI": "GOLD FIELDS", "SPGI": "S&P GLOBAL", "APD": "AIR PRODUCTS", "O": "REALTY INCOME", "HD": "HOME DEPOT",
    "JLL": "JONES LANG", "BBY": "BEST BUY", "NDAQ": "", "JHX": "JAMES HARDIE", "SJM": "SMUCKER",
    "RJF": "RAYMOND JAMES", "STLD": "STEEL DYNAMICS", "ELS": "EQUITY LIFESTYLE", "PFGC": "PERFORMANCE FOOD",
    "CORZ": "CORE SCIENTIFIC", "ELF": "E L F", "RRC": "RANGE RESOURCES", "CHRW": "H ROBINSON",
    "BMY": "BRISTOL MYERS", "EPD": "ENTERPRISE PRODUCTS", "BOX": "", "BROS": "DUTCH BROS", "PLUG": "PLUG POWER",
    "MTCH": "MATCH GROUP", "GAP": "", "XYZ": "", "SNAP": "", "RBC": "RBC BEARINGS", "JD": "JD COM",
    # Palabra clave compartida por dos empresas del universo
    "BAC": "BANK OF AMERICA", "BMO": "BANK OF MONTREAL", "BNS": "BANK OF NOVA", "BNY": "MELLON",
    "USB": "US BANCORP", "USFD": "US FOODS", "ACHR": "ARCHER AVIATION", "ADM": "ARCHER DANIELS",
    "BIO": "BIO RAD", "TECH": "BIO TECHNE", "BAM": "BROOKFIELD ASSET", "KO": "COCA COLA",
    "CCEP": "COCA COLA EUROPACIFIC", "STZ": "CONSTELLATION BRANDS", "CEG": "CONSTELLATION ENERGY",
    "CCK": "CROWN HOLDINGS", "CCI": "CROWN CASTLE", "DLTR": "DOLLAR TREE", "DG": "DOLLAR GENERAL",
    "FDXF": "FEDEX FREIGHT", "FIS": "FIDELITY NATIONAL", "FNF": "FIDELITY NATIONAL FINANCIAL",
    "GEV": "GE VERNOVA", "GEHC": "GE HEALTHCARE", "HONA": "HONEYWELL AEROSPACE", "HII": "HUNTINGTON INGALLS",
    "HBAN": "HUNTINGTON BANCSHARES", "ISRG": "INTUITIVE SURGICAL", "LUNR": "INTUITIVE MACHINES",
    "JNJ": "JOHNSON & JOHNSON", "JCI": "JOHNSON CONTROLS", "ODFL": "OLD DOMINION", "ONB": "OLD NATIONAL",
    "ON": "ON SEMICONDUCTOR", "ONON": "ON HOLDING", "PNFP": "PINNACLE FINANCIAL", "PNW": "PINNACLE WEST",
    "PL": "PLANET LABS", "PLNT": "PLANET FITNESS", "RKLB": "ROCKET LAB", "RKT": "ROCKET COMPANIES",
    "SNA": "SNAP ON", "PSLV": "SPROTT PHYSICAL SILVER", "PHYS": "SPROTT PHYSICAL GOLD", "WPC": "W P CAREY",
    "GWW": "GRAINGER", "WMB": "WILLIAMS COMPANIES", "WSM": "WILLIAMS SONOMA",
}
# Tickers que en los titulares suelen ser otra cosa: RBC (Royal Bank of Canada), RS (el "RS Rating" de IBD)
TICKERS_AMBIGUOS = {"RBC", "RS"}


def normalizar_para_buscar(texto: str) -> str:
    """Texto en mayúsculas, con la puntuación (salvo &) convertida en espacios: 'Bio-Techne' -> 'BIO TECHNE'."""
    return " ".join(re.sub(r"[^A-Za-z0-9& ]", " ", texto).upper().split())


def palabra_clave_empresa(ticker: str, nombre: str) -> str:
    """Palabra clave con la que se busca a la empresa en los titulares.

    La de PALABRA_CLAVE_A_MANO si la tiene; si no, la primera palabra del nombre,
    o las dos primeras si la primera es demasiado común.
    """
    if ticker in PALABRA_CLAVE_A_MANO:
        return PALABRA_CLAVE_A_MANO[ticker]
    palabras = normalizar_para_buscar(nombre).split()
    if palabras[0] in PALABRAS_DEMASIADO_COMUNES and len(palabras) > 1:
        return palabras[0] + " " + palabras[1]
    return palabras[0]


def menciona_empresa(titular: str, ticker: str, palabra_clave: str) -> bool:
    """True si el titular contiene el ticker o la palabra clave de la empresa.

    El ticker tiene que ir en mayúsculas y suelto: ni 'Plug' vale por PLUG, ni 'PG&E' por PG, ni 'U.S.'
    por U. Un ticker de una letra o ambiguo solo vale entre paréntesis o tras la bolsa: '(T)', 'NYSE: T'.
    """
    if len(ticker) == 1 or ticker in TICKERS_AMBIGUOS:
        patron_ticker = r"\(" + re.escape(ticker) + r"\)|(?i:NYSE|NASDAQ)\s*:\s*" + re.escape(ticker) + r"\b"
    else:
        patron_ticker = r"(?<![\w&.-])" + re.escape(ticker) + r"(?![\w&-]|\.\w)"
    if re.search(patron_ticker, titular):
        return True
    # Una palabra clave de 1-2 letras (p. ej. "X") aparecería en cualquier titular: solo vale el ticker
    return len(palabra_clave) >= 3 and re.search(r"\b" + re.escape(palabra_clave) + r"\b", normalizar_para_buscar(titular)) is not None


# --- NewsAPI (notebook 06 y app.py) ---
URL_NEWSAPI = "https://newsapi.org/v2/everything"
# Solo medios financieros: sin este filtro, buscar "Garmin" devuelve sorteos de relojes y paquetes de software
MEDIOS_FINANCIEROS = [
    "finance.yahoo.com", "marketwatch.com", "fool.com", "seekingalpha.com", "benzinga.com", "cnbc.com",
    "investors.com", "zacks.com", "barrons.com", "wsj.com", "bloomberg.com", "reuters.com", "forbes.com",
    "businesswire.com", "prnewswire.com", "globenewswire.com", "biztoc.com", "thestreet.com", "investing.com",
    "nasdaq.com", "businessinsider.com", "ft.com", "axios.com", "gurufocus.com", "insidermonkey.com", "247wallst.com",
]


def buscar_titulares_newsapi(palabras: list[str], desde: pd.Timestamp) -> list[dict]:
    """Busca en los titulares de medios financieros cualquiera de las palabras, desde la fecha dada.

    Requiere haber llamado antes a load_dotenv(). Devuelve la lista de artículos tal como la da
    NewsAPI. Lanza RuntimeError si falta la clave, si hay un fallo de red o timeout, o si la API
    responde con un error (p. ej. el límite diario agotado).
    """
    clave = os.getenv("NEWSAPI_KEY", "").strip()
    if not clave:
        raise RuntimeError("Falta NEWSAPI_KEY en el .env")
    parametros = {
        "q": " OR ".join(f'"{palabra}"' for palabra in palabras),
        "searchIn": "title",
        "domains": ",".join(MEDIOS_FINANCIEROS),
        "language": "en",
        "from": desde.strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "pageSize": 100,
    }
    try:
        respuesta = requests.get(URL_NEWSAPI, params=parametros, headers={"X-Api-Key": clave}, timeout=30)
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Fallo de red al consultar NewsAPI: {error}") from error

    if respuesta.status_code != 200:
        # Los errores de NewsAPI suelen venir en JSON con un mensaje; si no, se muestra el principio del texto
        try:
            detalle = respuesta.json().get("message", "")
        except ValueError:
            detalle = respuesta.text[:200]
        raise RuntimeError(f"NewsAPI respondió HTTP {respuesta.status_code}: {detalle}")
    return respuesta.json()["articles"]


# --- Evaluación de los modelos de sentimiento (notebooks 07 y 08) ---
CLASES_SENTIMIENTO = ("negative", "neutral", "positive")
# Remuestreos del bootstrap para el intervalo de confianza del F1 macro
BOOTSTRAP_REMUESTREOS = 1000


def evaluar_clasificacion(reales, predichas, clases: tuple[str, ...] = CLASES_SENTIMIENTO,
                          remuestreos: int = BOOTSTRAP_REMUESTREOS) -> dict:
    """Mide un clasificador de sentimiento comparando sus predicciones con las etiquetas reales.

    Recibe las etiquetas reales y las predichas (listas o series de la misma longitud, con
    valores de `clases`). Devuelve un diccionario con el F1 macro, su intervalo de confianza
    del 95 % por bootstrap (se repite el cálculo sobre `remuestreos` muestras con
    reemplazamiento de las mismas frases, con SEMILLA, y se toman los percentiles 2,5 y 97,5),
    la exactitud, la matriz de confusión (filas = real, columnas = predicha) y el número de frases.
    Lanza ValueError si las longitudes no coinciden, no hay frases o aparece una clase desconocida.
    """
    reales, predichas = np.asarray(reales), np.asarray(predichas)
    if len(reales) != len(predichas) or len(reales) == 0:
        raise ValueError(f"Hacen falta las mismas frases en reales ({len(reales)}) y predichas ({len(predichas)}), y al menos una")
    desconocidas = (set(reales) | set(predichas)) - set(clases)
    if desconocidas:
        raise ValueError(f"Clases desconocidas: {desconocidas} (se esperaban {clases})")

    # zero_division=0: una clase que el modelo nunca predice tiene F1 = 0 (sin aviso)
    def f1_macro(r, p):
        return f1_score(r, p, labels=list(clases), average="macro", zero_division=0)

    generador = np.random.default_rng(SEMILLA)
    f1_remuestreos = []
    for _ in range(remuestreos):
        indices = generador.integers(0, len(reales), size=len(reales))
        f1_remuestreos.append(f1_macro(reales[indices], predichas[indices]))
    ic_inferior, ic_superior = np.percentile(f1_remuestreos, [2.5, 97.5])

    matriz = pd.DataFrame(confusion_matrix(reales, predichas, labels=list(clases)),
                          index=pd.Index(clases, name="real"), columns=pd.Index(clases, name="predicha"))
    return {
        "f1_macro": float(f1_macro(reales, predichas)),
        "f1_ic_inferior": float(ic_inferior),
        "f1_ic_superior": float(ic_superior),
        "exactitud": float(accuracy_score(reales, predichas)),
        "matriz_confusion": matriz,
        "frases": len(reales),
    }


def comparar_f1(reales, predichas_a, predichas_b, clases: tuple[str, ...] = CLASES_SENTIMIENTO,
                remuestreos: int = BOOTSTRAP_REMUESTREOS) -> dict:
    """Diferencia de F1 macro entre dos modelos (a menos b) sobre las mismas frases, con su intervalo del 95 %.

    El intervalo sale de un bootstrap emparejado: en cada remuestreo (con SEMILLA) se toman las
    mismas frases para los dos modelos y se resta su F1 macro; se toman los percentiles 2,5 y 97,5.
    Si el intervalo contiene el 0, la diferencia no es concluyente.
    Lanza ValueError si las longitudes no coinciden o no hay frases.
    """
    reales, predichas_a, predichas_b = np.asarray(reales), np.asarray(predichas_a), np.asarray(predichas_b)
    if not len(reales) == len(predichas_a) == len(predichas_b) or len(reales) == 0:
        raise ValueError(f"Hacen falta las mismas frases en reales ({len(reales)}) y en los dos modelos "
                         f"({len(predichas_a)} y {len(predichas_b)}), y al menos una")

    def f1_macro(r, p):
        return f1_score(r, p, labels=list(clases), average="macro", zero_division=0)

    generador = np.random.default_rng(SEMILLA)
    diferencias = []
    for _ in range(remuestreos):
        indices = generador.integers(0, len(reales), size=len(reales))
        diferencias.append(f1_macro(reales[indices], predichas_a[indices]) - f1_macro(reales[indices], predichas_b[indices]))
    ic_inferior, ic_superior = np.percentile(diferencias, [2.5, 97.5])
    return {
        "diferencia": float(f1_macro(reales, predichas_a) - f1_macro(reales, predichas_b)),
        "ic_inferior": float(ic_inferior),
        "ic_superior": float(ic_superior),
    }



def tabla_f1(evaluaciones: dict) -> pd.DataFrame:
    """Tabla con el F1 macro, su intervalo del 95 % y la exactitud de cada evaluación, con 3 decimales.

    Recibe un diccionario {nombre: resultado de evaluar_clasificacion}; los nombres pasan a ser las filas.
    Lanza ValueError si está vacío.
    """
    if not evaluaciones:
        raise ValueError("No hay ninguna evaluación que poner en la tabla")
    return pd.DataFrame({
        nombre: {"F1 macro": e["f1_macro"], "IC 95 % inferior": e["f1_ic_inferior"],
                 "IC 95 % superior": e["f1_ic_superior"], "exactitud": e["exactitud"]}
        for nombre, e in evaluaciones.items()
    }).T.round(3)


def tabla_diferencias(frases: pd.DataFrame, preguntas: dict, columna_real: str = "etiqueta") -> pd.DataFrame:
    """Diferencia de F1 macro de cada pregunta (comparar_f1), con su intervalo y si el 0 queda fuera de él.

    `frases` tiene la etiqueta real (`columna_real`) y una columna por modelo con sus predicciones;
    `preguntas` es {pregunta: (modelo_a, modelo_b)}, y la diferencia es a menos b.
    Lanza ValueError si falta la columna de algún modelo.
    """
    faltan = {modelo for par in preguntas.values() for modelo in par} - set(frases.columns)
    if faltan:
        raise ValueError(f"Faltan las predicciones de: {sorted(faltan)}")
    filas = {}
    for pregunta, (modelo_a, modelo_b) in preguntas.items():
        d = comparar_f1(frases[columna_real], frases[modelo_a], frases[modelo_b])
        filas[pregunta] = {"diferencia": round(d["diferencia"], 3), "IC 95 % inferior": round(d["ic_inferior"], 3),
                           "IC 95 % superior": round(d["ic_superior"], 3),
                           "¿concluyente?": "sí" if d["ic_inferior"] > 0 or d["ic_superior"] < 0 else "no (el 0 está dentro)"}
    return pd.DataFrame(filas).T



# --- Noticias y bolsa (notebooks 08 y 09) ---
def diferencia_medias_welch(grupo_a, grupo_b, confianza: float = 0.95) -> dict:
    """Diferencia de medias (a menos b) con su intervalo de confianza de Welch y el p-valor.

    Welch no supone que los dos grupos tengan la misma varianza. Supone observaciones independientes:
    con eventos del mismo día o de la misma empresa, el intervalo sale algo más estrecho de lo que debería.
    Lanza ValueError si algún grupo tiene menos de 2 valores.
    """
    grupo_a, grupo_b = np.asarray(grupo_a, dtype=float), np.asarray(grupo_b, dtype=float)
    if len(grupo_a) < 2 or len(grupo_b) < 2:
        raise ValueError(f"Hacen falta al menos 2 valores por grupo (hay {len(grupo_a)} y {len(grupo_b)})")
    prueba = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
    intervalo = prueba.confidence_interval(confianza)
    return {"diferencia": float(grupo_a.mean() - grupo_b.mean()), "ic_inferior": float(intervalo.low),
            "ic_superior": float(intervalo.high), "p_valor": float(prueba.pvalue), "n_a": len(grupo_a), "n_b": len(grupo_b)}


def media_con_intervalo(valores, confianza: float = 0.95) -> dict:
    """Media de unos valores con su intervalo de confianza (t de Student) y cuántos son.

    Lanza ValueError si hay menos de 2 valores.
    """
    valores = np.asarray(valores, dtype=float)
    if len(valores) < 2:
        raise ValueError(f"Hacen falta al menos 2 valores (hay {len(valores)})")
    media = valores.mean()
    mitad = stats.t.ppf((1 + confianza) / 2, len(valores) - 1) * valores.std(ddof=1) / np.sqrt(len(valores))
    return {"media": float(media), "ic_inferior": float(media - mitad), "ic_superior": float(media + mitad), "n": len(valores)}

# --- M1: diccionario Loughran-McDonald (notebooks 07 y 08) ---


def tono_loughran_mcdonald(texto: str, palabras_negativas: set[str], palabras_positivas: set[str]) -> float:
    """Tono de un texto según el diccionario: (palabras positivas - negativas) / palabras del texto.

    Recibe el texto y las dos listas de palabras en mayúsculas. Las palabras del texto son
    los trozos de solo letras, pasados a mayúsculas (los números y signos no cuentan).
    Devuelve un número entre -1 y 1; 0 si el texto no tiene ninguna palabra.
    """
    palabras = re.findall(r"[A-Z]+", texto.upper())
    if not palabras:
        return 0.0
    positivas = sum(palabra in palabras_positivas for palabra in palabras)
    negativas = sum(palabra in palabras_negativas for palabra in palabras)
    return (positivas - negativas) / len(palabras)


def clasificar_tono(tono: float, umbral: float) -> str:
    """Convierte un tono en clase: 'positive' si supera el umbral, 'negative' si baja de -umbral y 'neutral' si no.

    Lanza ValueError si el umbral es negativo.
    """
    if umbral < 0:
        raise ValueError(f"El umbral debe ser 0 o positivo, se recibió {umbral}")
    if tono > umbral:
        return "positive"
    if tono < -umbral:
        return "negative"
    return "neutral"
