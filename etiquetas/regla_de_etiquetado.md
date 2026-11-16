# Regla de etiquetado de los titulares a mano

Fijada el 16 nov 2026, antes de empezar a etiquetar.

**Pregunta para cada titular** (la misma que se hizo a los anotadores de Financial PhraseBank):

> Desde el punto de vista de un inversor en esa empresa, ¿este titular es una noticia **buena** (`positive`), **mala** (`negative`) o **neutra** (`neutral`) para el precio de su acción?

**Condiciones:**

1. Solo con el titular, que es lo único que ven los modelos. Sin abrir el enlace ni buscar nada más.
2. La empresa es la de las columnas `ticker` y `empresa`. Si el titular habla de varias, se juzga para esa.
3. Si el titular **no trata de esa empresa** (solo coincide una palabra, como *principal*, *hinge* o *holding*), `trata_de_la_empresa` = `no` y `etiqueta` = `neutral`. Si trata de ella, `trata_de_la_empresa` = `sí`.
4. En la duda, `neutral`.
5. Sin mirar lo que dicen los modelos.

**Formato:** en la columna `trata_de_la_empresa`, `sí` o `no`; en la columna `etiqueta`, una de estas tres palabras: `positive`, `negative` o `neutral`. No se cambia ninguna otra columna.
