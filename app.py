"""¿Mueven las noticias la bolsa? App del proyecto: los resultados y un explorador por empresa.

Se abre con `streamlit run app.py`, después de ejecutar los notebooks. Solo lee lo que han guardado en data/cache/:
no descarga, no entrena y no recalcula ningún resultado. El tema está en .streamlit/config.toml.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from paginas.estilo import BORDE, FUENTE, REJILLA, SECUNDARIO, TEXTO

st.set_page_config(page_title="¿Mueven las noticias la bolsa?", page_icon=":material/newspaper:", layout="wide")

# CSS mínimo, sin clases internas de Streamlit: títulos algo más compactos y números del mismo ancho (tabulares)
st.markdown("""<style>
h1, h2, h3, h4 { letter-spacing: -0.015em; }
html, body { font-variant-numeric: tabular-nums; }
</style>""", unsafe_allow_html=True)

# Plantilla común de los gráficos: fondo transparente, rejilla suave, fuente del sistema y leyenda corta
eje = {"gridcolor": REJILLA, "zeroline": False, "linecolor": BORDE, "tickfont": {"color": SECUNDARIO}}
pio.templates["proyecto"] = go.layout.Template(layout={
    "font": {"family": FUENTE, "color": TEXTO, "size": 13},
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "xaxis": eje, "yaxis": eje, "hoverlabel": {"font": {"family": FUENTE}},
    "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom", "title": None, "font": {"size": 12}},
    "margin": {"t": 30, "l": 10, "r": 10, "b": 10},
    "separators": ",.",   # coma decimal y punto de miles, en los ejes y al pasar el ratón
})
pio.templates.default = "plotly_white+proyecto"

pagina = st.navigation([
    st.Page("paginas/proyecto.py", title="Proyecto", icon=":material/info:", default=True),
    st.Page("paginas/modelos.py", title="Modelos de sentimiento", icon=":material/psychology:"),
    st.Page("paginas/noticias_y_bolsa.py", title="Noticias y bolsa", icon=":material/show_chart:"),
    st.Page("paginas/explorador.py", title="Explorador", icon=":material/search:"),
])
pagina.run()
