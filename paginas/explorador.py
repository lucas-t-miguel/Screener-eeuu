"""Página «Explorador»: precios, señales técnicas y noticias de una empresa, de octubre de 2016 a junio de 2020.

Lee los datos diarios que guardaron los notebooks 02, 04, 05 y 08; no calcula indicadores nuevos ni carga ningún modelo.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from funciones import CARPETA_CACHE
from paginas.estilo import ACENTO, NEGATIVO, NEUTRAL, POSITIVO, SECUNDARIO, nota_aviso

INICIO, FIN = pd.Timestamp("2016-10-10"), pd.Timestamp("2020-06-11")   # el periodo con precios y noticias de FNSPID
MODELO = "M4 FinBERT"                                                    # el mejor modelo de sentimiento
SENTIMIENTO = {1: ("positivo", POSITIVO), 0: ("neutral", NEUTRAL), -1: ("negativo", NEGATIVO)}
TITULARES_EN_LA_ETIQUETA = 5


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

st.title("Explorador")
con_noticias = sorted(titulares["ticker"].unique())
eleccion_empresa, eleccion_fechas = st.columns([3, 2])
ticker = eleccion_empresa.selectbox("Empresa (las 633 con noticias)", con_noticias, index=con_noticias.index("AAPL"),
                                    format_func=lambda t: f"{t} — {empresas.loc[t, 'name']}")
fechas = eleccion_fechas.date_input("Fechas (de oct 2016 a jun 2020)", value=(FIN - timedelta(days=365), FIN),
                                    min_value=INICIO, max_value=FIN)
if len(fechas) != 2:
    st.stop()   # mientras se elige la segunda fecha
desde, hasta = pd.Timestamp(fechas[0]), pd.Timestamp(fechas[1])

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

st.subheader(f"{ticker} · {empresas.loc[ticker, 'name']}")
st.caption(f"Sector (Fama-French): {empresas.loc[ticker, 'sector_descripcion']} · del {desde:%d/%m/%Y} al {hasta:%d/%m/%Y}")
ultimo = dias.iloc[-1]
tarjetas = st.columns(5)
tarjetas[0].metric("Último cierre (ajustado)", f"{ultimo['close']:,.2f} USD", border=True)
tarjetas[1].metric("Régimen el último día", ultimo["regimen"] or "—", border=True)
tarjetas[2].metric("Señales de compra", int((dias["primera_senal"] == "compra").sum()), border=True)
tarjetas[3].metric("Señales de venta", int((dias["primera_senal"] == "venta").sum()), border=True)
tarjetas[4].metric("Titulares", len(noticias), border=True,
                   help=" · ".join(f"{(noticias['sentimiento'] == s).sum()} {nombre}s" for s, (nombre, _) in SENTIMIENTO.items()))

# --- Cuatro paneles con el mismo eje de fechas: velas, franja de noticias, RSI y ADX
grafico = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.05, 0.2, 0.2], vertical_spacing=0.025)
grafico.add_trace(go.Candlestick(x=dias["fecha"], open=dias["open"], high=dias["high"], low=dias["low"], close=dias["close"],
                                 name="precio", increasing_line_color=POSITIVO, decreasing_line_color=NEGATIVO,
                                 showlegend=False), row=1, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["ema_rapida"], name="EMA(4)", line={"width": 1.3, "color": ACENTO}), row=1, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["ema_lenta"], name="EMA(8)", line={"width": 1.3, "color": SECUNDARIO}), row=1, col=1)
banda = {"width": 1, "color": "rgba(0, 102, 204, 0.35)", "dash": "dot"}
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["donchian_maximo_previo"], name="Donchian", line=banda), row=1, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["donchian_minimo_previo"], name="Donchian", line=banda, showlegend=False,
                             fill="tonexty", fillcolor="rgba(0, 102, 204, 0.05)"), row=1, col=1)
for senal, simbolo, color, altura in [("compra", "triangle-up", POSITIVO, dias["low"] * 0.97),
                                      ("venta", "triangle-down", NEGATIVO, dias["high"] * 1.03)]:
    es_senal = dias["primera_senal"] == senal
    grafico.add_trace(go.Scatter(x=dias.loc[es_senal, "fecha"], y=altura[es_senal], mode="markers", name=senal,
                                 marker={"symbol": simbolo, "size": 10, "color": color}), row=1, col=1)

for signo, (nombre, color) in SENTIMIENTO.items():
    del_signo = por_dia[por_dia["signo"] == signo]
    grafico.add_trace(go.Scatter(
        x=del_signo["sesion"], y=[0] * len(del_signo), mode="markers", name=f"noticias {nombre}s",
        marker={"size": 7, "color": color, "symbol": "square"}, customdata=del_signo[["cuantos", "texto"]],
        hovertemplate="%{x|%d/%m/%Y} · %{customdata[0]} titular(es)<br>%{customdata[1]}<extra></extra>"), row=2, col=1)

grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["rsi"], name="RSI", line={"color": ACENTO, "width": 1.3}, showlegend=False), row=3, col=1)
for nivel in [30, 70]:
    grafico.add_hline(y=nivel, line_dash="dot", line_color=NEUTRAL, row=3, col=1)
grafico.add_trace(go.Scatter(x=dias["fecha"], y=dias["adx"], name="ADX", line={"color": SECUNDARIO, "width": 1.3}, showlegend=False), row=4, col=1)
grafico.add_hline(y=25, line_dash="dot", line_color=NEUTRAL, row=4, col=1)

grafico.update_layout(height=760, xaxis_rangeslider_visible=False)
grafico.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])   # sin los fines de semana
grafico.update_yaxes(title_text="precio (USD)", row=1, col=1)
grafico.update_yaxes(showticklabels=False, showgrid=False, range=[-1, 1], title_text="noticias", row=2, col=1)
grafico.update_yaxes(range=[0, 100], title_text="RSI (14)", row=3, col=1)
grafico.update_yaxes(title_text="ADX (14)", row=4, col=1)
st.plotly_chart(grafico, width="stretch", theme=None)

st.subheader(f"Titulares del periodo, según M4 ({len(noticias)})")
tabla = (noticias.sort_values("fecha", ascending=False)
         .assign(sentimiento=lambda t: t["sentimiento"].map(lambda s: SENTIMIENTO[s][0]))[["fecha", "titular", "sentimiento", "confianza"]])
st.dataframe(tabla, hide_index=True, width="stretch",
             column_config={"fecha": st.column_config.DateColumn(format="DD/MM/YYYY"),
                            "confianza": st.column_config.NumberColumn(format="%.2f", help="probabilidad de la clase elegida")})
st.caption("Señales: primer día de cada racha (notebook 05). Noticias: titulares de FNSPID con el sentimiento de M4 "
           "(FinBERT ajustado en el notebook 07); los de fin de semana van al viernes.")
nota_aviso()
