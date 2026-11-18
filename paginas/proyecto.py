"""Página «Proyecto»: la pregunta, los datos, el método y el aviso, en una pantalla."""

import streamlit as st

from paginas.estilo import nota_aviso

st.title("¿Mueven las noticias la bolsa?")
st.markdown(":gray[¿Se relaciona el sentimiento de las noticias de una empresa con lo que hace después su acción? "
            "Sentimiento de titulares financieros medido con cuatro modelos, y su relación con las acciones de EEUU.]")

st.subheader("Los datos")
columnas = st.columns(4)
columnas[0].metric("Empresas de NASDAQ y NYSE", "1.000", help="Las más líquidas, con 10 años de precios de yfinance", border=True)
columnas[1].metric("Titulares de noticias", "192.069", help="FNSPID: titulares que mencionan a su empresa, oct 2016 – jun 2020", border=True)
columnas[2].metric("Frases etiquetadas", "4.836", help="Financial PhraseBank: entrenar y examinar los modelos", border=True)
columnas[3].metric("Titulares etiquetados a mano", "300", help="Titulares recientes de NewsAPI, para medir fuera de PhraseBank", border=True)

st.subheader("El método, en 4 pasos")
pasos = [
    ("1 · Los datos", "Empresas y sectores de la SEC, precios de yfinance, titulares de FNSPID y NewsAPI, "
                      "e informes de la SEC como texto para preentrenar."),
    ("2 · Medir el sentimiento", "Cuatro modelos: un diccionario, un Transformer, un BERT preentrenado aquí y FinBERT. "
                                 "Examen con el test de PhraseBank y con 300 titulares etiquetados a mano."),
    ("3 · Noticias y bolsa", "Rentabilidad de la acción menos la del mercado (SPY) en la sesión siguiente a las noticias "
                             "positivas, neutrales y negativas."),
    ("4 · Señales técnicas", "¿Cambia lo que pasa después de una señal de compra o de venta según las noticias "
                             "de los días anteriores?"),
]
for columna, (titulo, texto) in zip(st.columns(4), pasos):
    with columna.container(border=True, height="stretch"):
        st.markdown(f"**{titulo}**")
        st.caption(texto)

st.markdown(":gray[Los resultados están en las páginas **Modelos de sentimiento** y **Noticias y bolsa**; "
            "el **Explorador** enseña los datos de cada empresa.]")
nota_aviso()
