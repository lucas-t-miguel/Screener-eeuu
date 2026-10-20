"""Funciones reutilizables del Screener Financiero EEUU.

Aquí solo hay definiciones de funciones y constantes: importar este
archivo no ejecuta nada. Cada función está aquí porque se usa en más de
un notebook (o en un notebook y en app.py).
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

# --- Rutas ---
CARPETA_CACHE = Path(__file__).resolve().parent / "data" / "cache"

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


# --- Indicadores técnicos (sección 8.2 del CLAUDE.md) ---
RSI_SESIONES = 14
MACD_RAPIDA, MACD_LENTA, MACD_SENAL = 12, 26, 9
BOLLINGER_SESIONES, BOLLINGER_DESVIACIONES = 20, 2
FIBONACCI_SESIONES = 45
FIBONACCI_NIVELES = (0.236, 0.382, 0.5, 0.618, 0.786)
CMF_SESIONES = 20
ADX_SESIONES = 14
EMA_RAPIDA, EMA_LENTA = 4, 8
DONCHIAN_SESIONES = 20
ATR_SESIONES = 14


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
    """Calcula todos los indicadores técnicos del proyecto para UN ticker.

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

    # MACD: diferencia entre dos medias exponenciales, su señal y el histograma
    macd = ta.macd(cierre, fast=MACD_RAPIDA, slow=MACD_LENTA, signal=MACD_SENAL)
    indicadores["macd"] = macd.iloc[:, 0]
    indicadores["macd_histograma"] = macd.iloc[:, 1]
    indicadores["macd_senal"] = macd.iloc[:, 2]

    # Bandas de Bollinger: media de 20 sesiones ± 2 desviaciones típicas
    bandas = ta.bbands(cierre, length=BOLLINGER_SESIONES, lower_std=BOLLINGER_DESVIACIONES,
                       upper_std=BOLLINGER_DESVIACIONES, ddof=0)
    indicadores["bollinger_inferior"] = bandas.iloc[:, 0]
    indicadores["bollinger_media"] = bandas.iloc[:, 1]
    indicadores["bollinger_superior"] = bandas.iloc[:, 2]

    # Fibonacci dinámico (a mano, no existe en pandas-ta): niveles medidos desde el máximo
    # de las últimas 45 sesiones hacia abajo, en proporción al recorrido máximo - mínimo
    maximo = alto.rolling(FIBONACCI_SESIONES).max()
    minimo = bajo.rolling(FIBONACCI_SESIONES).min()
    indicadores["fibonacci_maximo"] = maximo
    indicadores["fibonacci_minimo"] = minimo
    for nivel in FIBONACCI_NIVELES:
        indicadores[f"fibonacci_{nivel * 100:.1f}".replace(".", "_")] = maximo - nivel * (maximo - minimo)

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

    # ATR: rango medio diario, base de todos los stops
    indicadores["atr"] = ta.atr(alto, bajo, cierre, length=ATR_SESIONES)

    return indicadores


# --- Régimen de mercado y señales (sección 8.3 del CLAUDE.md) ---
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


# --- Riesgo y gestión de la operación (sección 8.4 del CLAUDE.md) ---
STOP_FIJO_ATR = 2
STOP_TRAILING_ATR = 3
RANGO_SIN_NIVEL_ATR = 2
# Un soporte/resistencia cuenta como "cercano" si está a como mucho 3 ATR del cierre
# (la misma anchura que el trailing-stop). Elegido en el notebook 05.
RANGO_NIVEL_CERCANO_ATR = 3
COLUMNAS_NIVELES_PRECIO = ("fibonacci_23_6", "fibonacci_38_2", "fibonacci_50_0", "fibonacci_61_8",
                           "fibonacci_78_6", "bollinger_inferior", "bollinger_media", "bollinger_superior")
SESIONES_POR_ANIO = 252


def calcular_stops_operacion(precios_operacion: pd.DataFrame, precio_entrada: float,
                             atr_entrada: float, sentido: str) -> pd.DataFrame:
    """Calcula los stops de UNA operación para cada sesión desde la entrada.

    Recibe los precios (high, low) desde el día de entrada incluido, ordenados por fecha,
    el precio de entrada, el ATR del día de la señal (congelado toda la operación) y el
    sentido ('compra' o 'venta'). Devuelve un DataFrame con stop_fijo, stop_trailing y
    stop_activo (el más protector de los dos). El stop de cada sesión solo usa los
    precios hasta la sesión anterior: es el nivel que se conoce al empezar el día.
    Lanza ValueError si el sentido no es válido o el ATR no es positivo.
    """
    if sentido not in ("compra", "venta"):
        raise ValueError(f"Sentido no válido: {sentido!r} (debe ser 'compra' o 'venta')")
    if not atr_entrada > 0:
        raise ValueError(f"El ATR de entrada debe ser positivo, se recibió {atr_entrada}")

    stops = pd.DataFrame(index=precios_operacion.index)
    if sentido == "compra":
        # Máximo alcanzado desde la entrada hasta AYER, en tres pasos:
        #   cummax()  -> máximo acumulado hasta hoy
        #   shift(1)  -> se retrasa un día para no usar el máximo de hoy (el primer día vale el precio de entrada)
        #   clip()    -> nunca por debajo del precio de entrada
        extremo = precios_operacion["high"].cummax().shift(1, fill_value=precio_entrada).clip(lower=precio_entrada)
        stops["stop_fijo"] = precio_entrada - STOP_FIJO_ATR * atr_entrada
        stops["stop_trailing"] = extremo - STOP_TRAILING_ATR * atr_entrada
        stops["stop_activo"] = stops[["stop_fijo", "stop_trailing"]].max(axis=1)
    else:
        # Espejo de la compra: mínimo alcanzado desde la entrada hasta AYER, nunca por encima de la entrada
        extremo = precios_operacion["low"].cummin().shift(1, fill_value=precio_entrada).clip(upper=precio_entrada)
        stops["stop_fijo"] = precio_entrada + STOP_FIJO_ATR * atr_entrada
        stops["stop_trailing"] = extremo + STOP_TRAILING_ATR * atr_entrada
        stops["stop_activo"] = stops[["stop_fijo", "stop_trailing"]].min(axis=1)
    return stops


def calcular_rango_esperado(cierre: pd.Series, indicadores: pd.DataFrame) -> pd.DataFrame:
    """Rango de precio esperado (solo informativo): soporte y resistencia más cercanos.

    Recibe el cierre y los indicadores (niveles de Fibonacci, Bollinger y atr) con el
    mismo índice; sirve para muchas filas a la vez. El soporte es el nivel más alto por
    debajo del cierre y la resistencia el más bajo por encima, si están a como mucho
    RANGO_NIVEL_CERCANO_ATR × ATR; si no hay nivel cercano por un lado, se usa el cierre
    ∓ 2 × ATR. Devuelve soporte, resistencia y de qué tipo es cada uno ('nivel' o 'atr').
    """
    niveles = indicadores[list(COLUMNAS_NIVELES_PRECIO)]
    atr = indicadores["atr"]
    distancia_maxima = RANGO_NIVEL_CERCANO_ATR * atr

    por_debajo = niveles.where(niveles.lt(cierre, axis=0) & niveles.ge(cierre - distancia_maxima, axis=0))
    por_encima = niveles.where(niveles.gt(cierre, axis=0) & niveles.le(cierre + distancia_maxima, axis=0))

    rango = pd.DataFrame(index=cierre.index)
    rango["soporte"] = por_debajo.max(axis=1)
    rango["resistencia"] = por_encima.min(axis=1)
    rango["tipo_soporte"] = np.where(rango["soporte"].notna(), "nivel", "atr")
    rango["tipo_resistencia"] = np.where(rango["resistencia"].notna(), "nivel", "atr")
    rango["soporte"] = rango["soporte"].fillna(cierre - RANGO_SIN_NIVEL_ATR * atr)
    rango["resistencia"] = rango["resistencia"].fillna(cierre + RANGO_SIN_NIVEL_ATR * atr)
    return rango


def calcular_volatilidad_y_sharpe(cierres: pd.Series) -> dict:
    """Volatilidad histórica anualizada y Sharpe Ratio (tasa libre de riesgo = 0 %).

    Recibe los cierres de UN ticker ordenados por fecha. Usa los retornos diarios:
    volatilidad = desviación típica × √252; Sharpe = media / desviación típica × √252.
    Devuelve un diccionario con volatilidad, sharpe, retorno_medio_anual y sesiones.
    Lanza ValueError si hay menos de 2 retornos o la desviación típica es 0.
    """
    retornos = cierres.pct_change().dropna()
    if len(retornos) < 2 or retornos.std() == 0:
        raise ValueError("Hacen falta al menos 2 retornos con alguna variación")
    raiz_anual = np.sqrt(SESIONES_POR_ANIO)
    return {
        "volatilidad": retornos.std() * raiz_anual,
        "sharpe": retornos.mean() / retornos.std() * raiz_anual,
        "retorno_medio_anual": retornos.mean() * SESIONES_POR_ANIO,
        "sesiones": len(retornos),
    }


# --- Estacionalidad (sección 8.5 del CLAUDE.md, informativa) ---
ESTACIONALIDAD_ANIOS = 5
NOMBRES_MESES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
NOMBRES_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes")


def calcular_estacionalidad(cierres: pd.Series, anios: int = ESTACIONALIDAD_ANIOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estacionalidad de UN ticker en los últimos `anios` años completos de datos.

    Recibe los cierres con índice de fechas ordenado. Solo usa meses completos (descarta
    el mes en curso). Devuelve dos tablas: por mes del año (retorno medio, % de años en
    positivo y número de años) y por día de la semana (retorno diario medio y número de días).
    Lanza ValueError si no hay datos en el periodo.
    """
    inicio = cierres.index.max() - pd.DateOffset(years=anios)
    # Se toma también el cierre previo al inicio para poder calcular el primer retorno
    cierres = cierres[cierres.index >= inicio - pd.DateOffset(months=1)]
    if cierres.empty:
        raise ValueError("No hay cierres en el periodo pedido")

    cierres_mensuales = cierres.resample("ME").last()
    retornos_mensuales = cierres_mensuales.pct_change()
    retornos_mensuales = retornos_mensuales[retornos_mensuales.index > inicio]
    # BMonthEnd(0) lleva la fecha al último día hábil de su mes. Si el último dato es anterior
    # a ese día, el mes todavía no ha terminado y su retorno se descarta.
    ultima_fecha = cierres.index.max()
    if ultima_fecha < ultima_fecha + pd.offsets.BMonthEnd(0):
        retornos_mensuales = retornos_mensuales.iloc[:-1]  # el mes en curso no está completo
    retornos_mensuales = retornos_mensuales.dropna()
    # Se agrupa por número de mes (1-12) y después se cambia el número por su nombre
    por_mes = retornos_mensuales.groupby(retornos_mensuales.index.month).agg(
        retorno_medio="mean", porcentaje_anios_positivos=lambda r: (r > 0).mean(), anios="count")
    por_mes.index = [NOMBRES_MESES[mes - 1] for mes in por_mes.index]

    retornos_diarios = cierres.pct_change()
    retornos_diarios = retornos_diarios[retornos_diarios.index >= inicio].dropna()
    por_dia = retornos_diarios.groupby(retornos_diarios.index.dayofweek).agg(retorno_medio="mean", dias="count")
    por_dia.index = [NOMBRES_DIAS[dia] for dia in por_dia.index]
    return por_mes, por_dia


# --- Noticias: a qué empresa se refiere un titular (notebook 06 y app.py) ---
# Si la primera palabra del nombre es una de estas, se usan las dos primeras (TEXAS -> TEXAS INSTRUMENTS)
PALABRAS_DEMASIADO_COMUNES = {
    "AMERICAN", "GENERAL", "FIRST", "UNITED", "NATIONAL", "INTERNATIONAL", "TEXAS", "WESTERN", "EASTERN",
    "SOUTHERN", "NORTHERN", "GLOBAL", "THE", "NEW", "CENTRAL", "PACIFIC", "ATLANTIC", "ADVANCED", "APPLIED",
    "DIGITAL", "PUBLIC", "ROYAL", "STANDARD", "UNIVERSAL", "CAPITAL", "BANK", "ENERGY", "HEALTH", "CHINA",
    "CANADIAN", "PACKAGING", "ALLIED", "CONSOLIDATED", "WASTE",
}


def palabra_clave_empresa(nombre: str) -> str:
    """Primera palabra del nombre en mayúsculas, o las dos primeras si la primera es demasiado común."""
    palabras = re.sub(r"[^A-Za-z0-9& ]", " ", nombre).upper().split()
    if palabras[0] in PALABRAS_DEMASIADO_COMUNES and len(palabras) > 1:
        return palabras[0] + " " + palabras[1]
    return palabras[0]


def menciona_empresa(titular: str, ticker: str, palabra_clave: str) -> bool:
    """True si el titular contiene el ticker o la palabra clave como palabras sueltas."""
    titular = titular.upper()
    if re.search(r"\b" + re.escape(ticker) + r"\b", titular):
        return True
    # Una palabra clave de 1-2 letras (p. ej. "X") aparecería en cualquier titular: solo vale el ticker
    return len(palabra_clave) >= 3 and re.search(r"\b" + re.escape(palabra_clave) + r"\b", titular) is not None


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
    NewsAPI. Lanza RuntimeError si falta la clave o si la API responde con un error (p. ej. el
    límite diario agotado).
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
    respuesta = requests.get(URL_NEWSAPI, params=parametros, headers={"X-Api-Key": clave}, timeout=30)
    datos = respuesta.json()
    if respuesta.status_code != 200:
        raise RuntimeError(f"NewsAPI respondió {respuesta.status_code}: {datos.get('code')} - {datos.get('message')}")
    return datos["articles"]
