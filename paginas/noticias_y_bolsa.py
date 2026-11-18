"""Página «Noticias y bolsa»: el estudio de eventos del notebook 08 y las señales técnicas del 09."""

import plotly.graph_objects as go
import streamlit as st

from paginas.estilo import MODELOS, NEGATIVO, NEUTRAL, POSITIVO, con_intervalo, entero, leer_resultados, numero, nota_aviso

st.title("¿Se mueve la acción después de una noticia?")
st.markdown(":gray[Rentabilidad de la acción menos la del mercado (SPY), después de los días con noticias positivas "
            "y negativas de la empresa, según cada modelo. Unos 111.000 eventos (empresa y sesión), de 2016 a 2020.]")

eventos = leer_resultados("08_eventos").set_index(["momento", "modelo"])
siguiente = eventos.loc["la sesión siguiente"].loc[MODELOS[1:]]
mismo_dia = eventos.loc["el mismo día"].loc[MODELOS[1:]]

izquierda, derecha = st.columns([3, 2])
with izquierda:
    st.subheader("La sesión siguiente")
    # Verde si el intervalo queda por encima de 0, rojo si queda por debajo, gris si contiene el 0
    colores = [POSITIVO if inferior > 0 else NEGATIVO if superior < 0 else NEUTRAL
               for inferior, superior in zip(siguiente["IC 95 % inferior"], siguiente["IC 95 % superior"])]
    diferencia = siguiente["positivos − negativos (%)"]
    grafico = go.Figure(go.Scatter(
        x=diferencia, y=siguiente.index, mode="markers", marker={"size": 12, "color": colores}, showlegend=False,
        error_x={"type": "data", "symmetric": False, "array": siguiente["IC 95 % superior"] - diferencia,
                 "arrayminus": diferencia - siguiente["IC 95 % inferior"], "color": NEUTRAL, "thickness": 1.5, "width": 4},
        hovertemplate="%{y}: %{x:+.2f} puntos<extra></extra>"))
    grafico.add_vline(x=0, line_color=NEUTRAL, line_dash="dot")
    grafico.update_layout(height=300, xaxis={"title": "positivos − negativos (puntos porcentuales, IC 95 %)"},
                          yaxis={"categoryorder": "array", "categoryarray": MODELOS[:0:-1]})
    st.plotly_chart(grafico, width="stretch", theme=None)
with derecha:
    st.subheader("El mismo día")
    st.caption("Relación **inflada**: muchos titulares cuentan un movimiento que ya ha ocurrido "
               "(«las acciones caen tras los resultados»). No indica que la noticia anticipe nada.")
    for modelo, fila in mismo_dia.iterrows():
        st.markdown(f"{modelo}: **{numero(fila['positivos − negativos (%)'], 2)} puntos** "
                    f":gray[({numero(fila['IC 95 % inferior'], 2)} – {numero(fila['IC 95 % superior'], 2)})]")

st.subheader("¿Cambia una señal técnica según las noticias previas?")
st.markdown(":gray[Compras del análisis técnico (notebook 09), según las noticias de las 5 sesiones anteriores medidas "
            "con M4. Rentabilidad frente al mercado: positiva = la compra acierta.]")
senales = leer_resultados("09_senales")
compras = senales[senales["senal"] == "compra"]
tabla = compras.assign(valor=[con_intervalo(m, i, s, 2) for m, i, s in
                              zip(compras["media (%)"], compras["IC 95 % inferior"], compras["IC 95 % superior"])])
tabla = tabla.pivot(index="grupo", columns="horizonte", values="valor").loc[compras["grupo"].unique(), ["5 sesiones", "10 sesiones"]]
comparaciones = leer_resultados("09_comparaciones")
principal = comparaciones[(comparaciones["version"] == "principal") & (comparaciones["senal"] == "compra")]
for comparacion, filas in principal.groupby("comparacion", sort=False):
    tabla.loc[comparacion] = [con_intervalo(f["diferencia (%)"], f["IC 95 % inferior"], f["IC 95 % superior"], 2)
                              + ("" if f["¿concluyente?"] == "sí" else "  ·  no concluyente")
                              for _, f in filas.set_index("horizonte").loc[tabla.columns].iterrows()]
st.table(tabla.rename_axis("compras").rename(columns=lambda h: f"a {h} (%, IC 95 %)"))
senales_por_grupo = compras[compras["horizonte"] == "5 sesiones"].set_index("grupo")["señales"]
st.caption(f"Señales: {', '.join(f'{grupo} {entero(n)}' for grupo, n in senales_por_grupo.items())}.")

st.markdown("""**Lectura**
- **Hay relación, pero es pequeña, y solo la encuentran los modelos que leen bien los titulares** (M4 y el diccionario); con M2 y M3 no hay diferencia.
- **El mismo día la diferencia es mucho mayor**, pero sobre todo porque el titular cuenta lo que ya ha pasado.
- **Las noticias sirven como aviso, no como confirmación:** una compra técnica después de noticias negativas sale peor; las positivas no la mejoran.
- Sin la hora de publicación, esto no demuestra que se pueda anticipar la bolsa con las noticias.""")
nota_aviso()
