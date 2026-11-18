"""Página «Modelos de sentimiento»: el F1 macro de los cuatro modelos en PhraseBank y con los 300 titulares a mano."""

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from paginas.estilo import ACENTO, MODELOS, NEUTRAL, con_intervalo, leer_resultados, nota_aviso, numero

st.title("¿Qué modelo mide mejor el sentimiento?")
st.markdown(":gray[Cuatro formas de conseguir un modelo de sentimiento, medidas igual: F1 macro (la media del F1 "
            "de las tres clases) con su intervalo de confianza del 95 %. «Siempre neutral» es la referencia mínima.]")

modelos = leer_resultados("07_modelos")
a_mano = leer_resultados("08_titulares_a_mano")
conjuntos = {"Test de PhraseBank (968 frases)": (modelos[modelos["conjunto"] == "test"], ACENTO),
             "300 titulares etiquetados a mano": (a_mano[a_mano["conjunto"] == "los 300"], NEUTRAL)}

grafico = go.Figure()
for nombre, (tabla, color) in conjuntos.items():
    tabla = tabla.set_index("modelo").loc[MODELOS]
    grafico.add_trace(go.Scatter(
        x=tabla["f1_macro"], y=tabla.index, mode="markers", name=nombre, marker={"size": 11, "color": color},
        error_x={"type": "data", "symmetric": False, "array": tabla["ic_superior"] - tabla["f1_macro"],
                 "arrayminus": tabla["f1_macro"] - tabla["ic_inferior"], "color": color, "thickness": 1.5, "width": 4},
        hovertemplate="%{y}: %{x:.3f}<extra>" + nombre + "</extra>"))
grafico.update_layout(height=380, scattermode="group", scattergap=0.5, xaxis={"range": [0, 1], "title": "F1 macro (IC 95 %)"},
                      yaxis={"categoryorder": "array", "categoryarray": MODELOS[::-1]})
st.plotly_chart(grafico, width="stretch", theme=None)

# La tabla: cada modelo con sus dos exámenes
filas = []
for modelo in MODELOS:
    t = modelos[(modelos["conjunto"] == "test") & (modelos["modelo"] == modelo)].iloc[0]
    m300 = a_mano[(a_mano["conjunto"] == "los 300") & (a_mano["modelo"] == modelo)].iloc[0]
    m271 = a_mano[(a_mano["conjunto"] != "los 300") & (a_mano["modelo"] == modelo)].iloc[0]
    filas.append({"modelo": modelo,
                  "test de PhraseBank: F1 (IC 95 %)": con_intervalo(t["f1_macro"], t["ic_inferior"], t["ic_superior"]),
                  "exactitud": numero(t["exactitud"]),
                  "300 titulares: F1 (IC 95 %)": con_intervalo(m300["f1_macro"], m300["ic_inferior"], m300["ic_superior"]),
                  "los 271 que tratan de su empresa: F1": numero(m271["f1_macro"])})
st.table(pd.DataFrame(filas).set_index("modelo"))

def por_comparacion(tabla: pd.DataFrame, conjunto: str) -> pd.Series:
    """Cada diferencia de un conjunto, con su intervalo y si es concluyente, indexada por su código ('M2 − M1'…)."""
    tabla = tabla[tabla["conjunto"] == conjunto].copy()
    tabla["comparación"] = tabla["pregunta"].map(lambda pregunta: re.search(r"\((M\d − M\d)\)", pregunta).group(1))
    return tabla.set_index("comparación").apply(
        lambda f: con_intervalo(f["diferencia"], f["IC 95 % inferior"], f["IC 95 % superior"]) + f"  ·  {f['¿concluyente?']}", axis=1)


with st.expander("Diferencias entre modelos, con su intervalo (bootstrap emparejado)"):
    st.table(pd.DataFrame({"test de PhraseBank": por_comparacion(leer_resultados("07_diferencias"), "test"),
                           "300 titulares a mano": por_comparacion(leer_resultados("08_diferencias_a_mano"), "los 300")}))
    st.caption("M3 − M2 es la única comparación limpia del preentrenamiento. M4 − M2 y M4 − M3 comparan recetas "
               "completas: entre M3 y M4 cambian a la vez la red, el corpus, el tokenizador, la cabeza y el ajuste.")

st.markdown("""**Lectura**
- **Aprender de ejemplos supera al diccionario** en PhraseBank, pero no fuera de él: con los titulares, M2 y M3 casi se reducen a la mitad.
- **Preentrenar un modelo propio y pequeño (M3) no aporta**; partir de un BERT financiero preentrenado por otros (M4) es lo mejor en los dos exámenes.
- **M4 es el instrumento** con el que se miden las noticias en las páginas siguientes.""")
nota_aviso()
