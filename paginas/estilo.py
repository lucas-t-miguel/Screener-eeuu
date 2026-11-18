"""Lo que comparten las páginas de la app: la paleta, la nota del aviso y la lectura de los resultados."""

import pandas as pd
import streamlit as st

from funciones import CARPETA_CACHE

FUENTE = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
TEXTO, SECUNDARIO, ACENTO = "#1c1c1e", "#6e6e73", "#0066cc"
POSITIVO, NEGATIVO, NEUTRAL = "#1c8a4b", "#c1362d", "#8e8e93"   # también compra y venta
REJILLA, BORDE = "rgba(0, 0, 0, 0.06)", "rgba(0, 0, 0, 0.08)"
MODELOS = ["siempre neutral", "M1 diccionario", "M2 Transformer", "M3 BERT", "M4 FinBERT"]


@st.cache_data
def leer_resultados(nombre: str) -> pd.DataFrame:
    """Una tabla de resultados guardada por los notebooks (data/cache/resultados_<nombre>.parquet)."""
    ruta = CARPETA_CACHE / f"resultados_{nombre}.parquet"
    if not ruta.exists():
        st.error(f"Falta {ruta.name}: ejecuta antes los notebooks 07, 08 y 09.")
        st.stop()
    return pd.read_parquet(ruta)


def numero(valor: float, decimales: int = 3) -> str:
    """Un número con coma decimal, como en el resto del proyecto."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def entero(valor: int) -> str:
    """Un entero con punto de miles: '1.228'."""
    return f"{valor:,}".replace(",", ".")


def con_intervalo(valor: float, inferior: float, superior: float, decimales: int = 3) -> str:
    """'0,813 (0,782 – 0,841)'."""
    return f"{numero(valor, decimales)} ({numero(inferior, decimales)} – {numero(superior, decimales)})"


def nota_aviso() -> None:
    """La nota de cada página: discreta, pero visible."""
    st.divider()
    st.caption(":material/info: Proyecto académico de ciencia de datos. "
               "**Nada de lo que muestra esta app es una recomendación de inversión.**")
