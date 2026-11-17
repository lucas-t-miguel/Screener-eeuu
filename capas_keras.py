"""Capas y vocabulario de los modelos M2 y M3, y la construcción de M4, para cargarlos fuera del notebook 07.

Es una copia del código del notebook 07: secciones 5 y 6 (M2 y M3, que siguen los apuntes de clase,
`Transformers.ipynb` y `bert_simplificado.ipynb`) y sección 11 (M4, FinBERT). El 07 conserva sus propias
definiciones para no tener que volver a entrenar; este archivo solo lo importa el notebook 08.

Va aparte de `funciones.py` porque necesita Keras: importarlo fija el motor de Keras en TensorFlow,
el del proyecto, y por eso tiene que importarse antes que cualquier `import keras`.
"""

import os
import re
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"   # el motor de Keras del proyecto; se fija antes de importar keras

import keras
import keras_hub
import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization

if keras.backend.backend() != "tensorflow":
    raise ImportError(f"Keras ya estaba cargado con el motor '{keras.backend.backend()}': "
                      "importa capas_keras antes que keras (o reinicia el kernel)")

MAX_LEN = 64          # palabras por frase (se corta o se rellena), como en el notebook 07
VOCAB_SIZE = 30000    # tamaño del vocabulario, como en los apuntes


def custom_standardization(input_data):
    """Minúsculas, sin <br /> y sin signos de puntuación (como en los apuntes)."""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\\^_`{|}~"), ""
    )


def cargar_vectorizador(ruta_vocabulario: Path) -> TextVectorization:
    """La capa TextVectorization del notebook 07, con el vocabulario guardado en `ruta_vocabulario`.

    Lanza FileNotFoundError si el vocabulario no existe (se crea en el notebook 07).
    """
    if not ruta_vocabulario.exists():
        raise FileNotFoundError(f"No existe {ruta_vocabulario}: ejecuta antes el notebook 07")
    vectorize_layer = TextVectorization(max_tokens=VOCAB_SIZE, output_mode="int",
                                        standardize=custom_standardization, output_sequence_length=MAX_LEN)
    vectorize_layer.set_vocabulary(ruta_vocabulario.read_text(encoding="utf-8").splitlines())
    return vectorize_layer


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):  # Aquí se construye el Transformer
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    # sirve para serializar, lo que hace posible hacer save y load del modelo
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)   # Diferencia con los apuntes: tf.math.not_equal no vale en Keras 3

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


CAPAS_PROPIAS = {"TransformerEncoder": TransformerEncoder, "PositionalEmbedding": PositionalEmbedding}   # para cargar los .keras


# --- M4: FinBERT (sección 11 del notebook 07) ---
SECUENCIA_FINBERT = 136   # tokens por frase con los que se ajustó M4, con [CLS] y [SEP]


def cargar_tokenizador_finbert(carpeta_finbert: Path) -> keras_hub.tokenizers.BertTokenizer:
    """El tokenizador de FinBERT (FinVocab, en minúsculas), con el vocabulario de `carpeta_finbert`."""
    ruta_vocabulario = carpeta_finbert / "vocab.txt"
    if not ruta_vocabulario.exists():
        raise FileNotFoundError(f"No existe {ruta_vocabulario}: ejecuta antes el notebook 07")
    return keras_hub.tokenizers.BertTokenizer(vocabulary=str(ruta_vocabulario), lowercase=True)


def cargar_m4(carpeta_finbert: Path, ruta_pesos: Path) -> keras.Model:
    """M4 como en el notebook 07 (FinBERT con la GELU exacta y la cabeza [CLS]), con los pesos ajustados allí.

    Lanza FileNotFoundError si falta FinBERT convertido o los pesos ajustados (se crean en el notebook 07).
    """
    for ruta in [carpeta_finbert / "model.safetensors", ruta_pesos]:
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}: ejecuta antes el notebook 07")
    backbone = keras_hub.models.BertBackbone.from_preset(str(carpeta_finbert))
    # GELU exacta, la del BERT original con la que se preentrenó FinBERT (KerasHub usa una aproximación)
    for capa in backbone.transformer_layers:
        capa._feedforward_intermediate_dense.activation = keras.activations.gelu
    model = keras_hub.models.BertTextClassifier(backbone, num_classes=3, activation="softmax", dropout=0.1)
    model.load_weights(ruta_pesos)   # el fichero trae también el estado del optimizador del 07, que aquí no se usa
    return model
