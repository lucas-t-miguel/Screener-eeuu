"""Explorador del proyecto: precios, señales técnicas y noticias de una empresa, de octubre de 2016 a junio de 2020.

Se abre con `streamlit run app.py`. Solo lee lo que han guardado los notebooks en data/cache/: no descarga,
no calcula indicadores nuevos ni carga ningún modelo. Nada de lo que muestra es una recomendación de inversión.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from funciones import CARPETA_CACHE

INICIO, FIN = pd.Timestamp("2016-10-10"), pd.Timestamp("2020-06-11")   # el periodo con noticias de FNSPID y precios
MODELO = "M4 FinBERT"                                                    # el mejor modelo de sentimiento (notebooks 07 y 08)
NOMBRE_SENTIMIENTO = {1: "positivo", 0: "neutral", -1: "negativo"}
COLOR_SENTIMIENTO = {1: "#2ca02c", 0: "#7f7f7f", -1: "#d62728"}
TITULARES_EN_LA_ETIQUETA = 5                                             # al pasar el ratón por un día con noticias

st.set_page_config(page_title="Noticias y bolsa · explorador", layout="wide")


@st.cache_data
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Una tabla diaria (precios, indicadores y señales), los titulares con el sentimiento de M4, y nombre y sector."""
    def leer(fichero: str, columnas: list[str]) -> pd.DataFrame:
        tabla = pd.read_parquet(CARPETA_CACHE / f"{fichero}.parquet", columns=["fecha", "ticker"] + columnas)
        return tabla[tabla["fecha"].between(INICIO, FIN)]

    diario = (leer("precios_historicos", ["open", "high", "low", "close"])
              .merge(leer("indicadores", ["ema_rapida", "ema_lenta", "donchian_minimo_previo",
                                          "donchian_maximo_previo", "rsi", "adx"]), on=["fecha", "ticker"])
              .merge(leer("senales", ["senal", "regimen"]), on=["fecha", "ticker"])
              .sort_values(["ticker", "fecha"]).reset_index(drop=True))
    # Solo el primer día de cada racha de señales, como en el notebook 09
    diario["primera_senal"] = diario["senal"].where(diario["senal"] != diario.groupby("ticker")["senal"].shift())

    titulares = pd.read_parquet(CARPETA_CACHE / "sentimiento_titulares.parquet", filters=[("modelo", "==", MODELO)],
                                columns=["ticker", "fecha", "titular", "sentimiento", "confianza"])
    empresas = pd.read_parquet(CARPETA_CACHE / "empresas_sec.parquet", columns=["ticker", "name", "sector_descripcion"])
    return diario, titulares, empresas.set_index("ticker")


diario, titulares, empresas = cargar_datos()

# --- Barra lateral: empresa y fechas
st.sidebar.header("Qué ver")
con_noticias = sorted(titulares["ticker"].unique())
ticker = st.sidebar.selectbox("Empresa (las 633 con noticias)", con_noticias, index=con_noticias.index("AAPL"),
                              format_func=lambda t: f"{t} — {empresas.loc[t, 'name']}")
fechas = st.sidebar.date_input("Fechas (de oct 2016 a jun 2020)", value=(FIN - timedelta(days=365), FIN),
                               min_value=INICIO, max_value=FIN)
if len(fechas) != 2:
    st.stop()   # mientras se elige la segunda fecha
desde, hasta = pd.Timestamp(fechas[0]), pd.Timestamp(fechas[1])
st.sidebar.markdown("**Leyenda:** ▲ compra y ▼ venta (primer día de cada racha) · ● días con noticias, "
                    "del color de su sentimiento según M4 (verde positivo, gris neutral, rojo negativo)")

dias = diario[(diario["ticker"] == ticker) & diario["fecha"].between(desde, hasta)]
if dias.empty:
    st.warning("No hay sesiones de esta empresa en esas fechas.")
    st.stop()

# Cada titular va a la última sesión que no es posterior a él (los de fin de semana, al viernes), como en el notebook 08
sesiones = diario.loc[diario["ticker"] == ticker, ["fecha"]].rename(columns={"fecha": "sesion"})
noticias = pd.merge_asof(titulares[titulares["ticker"] == ticker].sort_values("fecha"), sesiones,
                         left_on="fecha", right_on="sesion")
noticias = noticias[noticias["sesion"].between(desde, hasta)]
por_dia = (noticias.groupby("sesion")
           .agg(media=("sentimiento", "mean"), cuantos=("titular", "size"),
                texto=("titular", lambda t: "<br>".join(t.str.slice(0, 90).head(TITULARES_EN_LA_ETIQUETA))
                       + ("<br>…" if len(t) > TITULARES_EN_LA_ETIQUETA else "")))
           .reset_index())
por_dia["signo"] = np.sign(por_dia["media"]).astype(int)

# --- Cabecera
st.title(f"{ticker} · {empresas.loc[ticker, 'name']}")
st.caption(f"Sector (Fama-French): {empresas.loc[ticker, 'sector_descripcion']} · del {desde:%d %b %Y} al {hasta:%d %b %Y}")
st.warning("Explorador de los datos del proyecto, con fines académicos. **No es una recomendación de inversión.**")

ultimo = dias.iloc[-1]
columnas = st.columns(5)
columnas[0].metric("Último cierre (ajustado)", f"{ultimo['close']:,.2f} USD")
columnas[1].metric("Régimen el último día", ultimo["regimen"] or "—")
columnas[2].metric("Señales de compra", int((dias["primera_senal"] == "compra").sum()))
columnas[3].metric("Señales de venta", int((dias["primera_senal"] == "venta").sum()))
columnas[4].metric("Titulares", len(noticias),
                   help=f"{(noticias['sentimiento'] == 1).sum()} positivos, {(noticias['sentimiento'] == 0).sum()} neutrales "
                        f"y {(noticias['sentimiento'] == -1).sum()} negativos, según M4")

# --- Gráfico: tres paneles con el mismo eje de fechas
grafico = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=["Precio, EMA(4) y EMA(8), canal de Donchian, señales y noticias", "RSI (14)", "ADX (14)"])
grafico.add_trace(go.Candlestick(x=dias["fecha"], open=dias["open"], high=dias["high"], low=dias["low"],
                                 close=dias["close"], name="precio"), row=1, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["ema_rapida"], name="EMA(4)", line={"width": 1.2, "color": "#ff7f0e"}), row=1, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["ema_lenta"], name="EMA(8)", line={"width": 1.2, "color": "#1f77b4"}), row=1, col=1)
banda = {"width": 1, "color": "rgba(120, 120, 200, 0.6)", "dash": "dot"}
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["donchian_maximo_previo"], name="Donchian (20 sesiones)", line=banda), row=1, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["donchian_minimo_previo"], name="Donchian (20 sesiones)", line=banda,
                             fill="tonexty", fillcolor="rgba(120, 120, 200, 0.08)", showlegend=False), row=1, col=1)

for senal, simbolo, color, altura in [("compra", "triangle-up", "#2ca02c", dias["low"] * 0.97),
                                      ("venta", "triangle-down", "#d62728", dias["high"] * 1.03)]:
    es_senal = dias["primera_senal"] == senal
    grafico.add_trace(go.Scatter(x=dias.loc[es_senal, "fecha"], y=altura[es_senal], mode="markers", name=f"señal de {senal}",
                                 marker={"symbol": simbolo, "size": 11, "color": color, "line": {"width": 1, "color": "black"}}),
                      row=1, col=1)

franja_de_noticias = dias["high"].max() * 1.08   # una fila de puntos por encima de las velas
for signo, dias_con_noticias in por_dia.groupby("signo"):
    grafico.add_trace(go.Scatter(
        x=dias_con_noticias["sesion"], y=[franja_de_noticias] * len(dias_con_noticias), mode="markers",
        name=f"noticias: {NOMBRE_SENTIMIENTO[signo]}", marker={"size": 8, "color": COLOR_SENTIMIENTO[signo]},
        customdata=dias_con_noticias[["cuantos", "texto"]],
        hovertemplate="%{x|%d %b %Y} · %{customdata[0]} titular(es)<br>%{customdata[1]}<extra></extra>"), row=1, col=1)

grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["rsi"], name="RSI", line={"color": "#9467bd"}, showlegend=False), row=2, col=1)
for nivel in [30, 70]:
    grafico.add_hline(y=nivel, line_dash="dot", line_color="gray", row=2, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["adx"], name="ADX", line={"color": "#8c564b"}, showlegend=False), row=3, col=1)
grafico.add_hline(y=25, line_dash="dot", line_color="gray", row=3, col=1)

grafico.update_layout(height=820, xaxis_rangeslider_visible=False, margin={"t": 40, "b": 20},
                      legend={"orientation": "h", "y": 1.06})
grafico.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])   # sin los fines de semana
grafico.update_yaxes(range=[0, 100], row=2, col=1)
st.plotly_chart(grafico, width="stretch")

# --- Tabla de titulares
st.subheader(f"Titulares del periodo, con su sentimiento según M4 ({len(noticias)})")
tabla = (noticias.sort_values("fecha", ascending=False)
         .assign(sentimiento=lambda t: t["sentimiento"].map(NOMBRE_SENTIMIENTO))[["fecha", "titular", "sentimiento", "confianza"]])
st.dataframe(tabla, hide_index=True, width="stretch",
             column_config={"fecha": st.column_config.DateColumn(format="YYYY-MM-DD"),
                            "confianza": st.column_config.NumberColumn(format="%.2f", help="probabilidad de la clase elegida")})
st.caption("Fuentes: precios de yfinance (ajustados), indicadores y señales de los notebooks 04 y 05, titulares de FNSPID "
           "y sentimiento de M4 (FinBERT ajustado en el notebook 07).")
