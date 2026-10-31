"""Lightweight CNN with dual (spatial + channel) attention for kidney CT classification.

Ported from ``notebooks/kidney-final.ipynb``. Three stacked conv blocks are
concatenated (dense-style feature reuse), refined by a residual spatial
attention module and a residual channel attention module, pooled, and
classified by a small MLP.
"""

from __future__ import annotations

from keras import Model, layers

BN_EPSILON = 1e-5

# Classifier heads: the one in the saved notebook (Dense 512 -> 256 -> 64) and a smaller one.
NOTEBOOK_DENSE_UNITS = (512, 256, 64)
COMPACT_DENSE_UNITS = (256, 64)


def residual_spatial_attention(x, reduction_ratio: int = 16):
    """Spatial attention map from 1x1 convs, applied multiplicatively with a residual add."""
    channels = x.shape[-1]
    h = layers.Conv2D(channels // reduction_ratio, 1, padding="same", activation="relu")(x)
    h = layers.Conv2D(channels // reduction_ratio, 1, padding="same", activation="relu")(h)
    h = layers.Conv2D(channels, 1, padding="same", activation="relu")(h)
    attention_map = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(h)
    return layers.Add()([layers.Multiply()([attention_map, x]), x])


def residual_channel_attention(x, gamma: int = 2):
    """Squeeze-and-excitation style channel attention with a residual add."""
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(channels // gamma, activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Add()([layers.Multiply()([x, s]), x])


def build_model(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    dense_units: tuple[int, ...] = NOTEBOOK_DENSE_UNITS,
) -> Model:
    """Build the dual-attention CNN.

    Args:
        input_shape: Input CT slice shape.
        num_classes: 4 (cyst, normal, stone, tumor).
        dense_units: Hidden MLP widths. The default reproduces the saved notebook
            (``NOTEBOOK_DENSE_UNITS``, 414,837 parameters). ``COMPACT_DENSE_UNITS``
            drops the first dense layer (225,909 parameters); it has not been
            trained here.
    """
    inputs = layers.Input(input_shape)

    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x1 = layers.BatchNormalization(axis=-1, epsilon=BN_EPSILON)(x1)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(x1)
    x2 = layers.BatchNormalization(axis=-1, epsilon=BN_EPSILON)(x2)
    x3 = layers.Conv2D(128, 3, padding="same", activation="relu")(x2)
    x3 = layers.BatchNormalization(axis=-1, epsilon=BN_EPSILON)(x3)
    features = layers.Concatenate()([x1, x2, x3])  # 224 channels

    features = residual_spatial_attention(features)
    features = residual_channel_attention(features)

    x = layers.GlobalAveragePooling2D()(features)
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="kidney_dual_attention_cnn")


def compile_model(model: Model, learning_rate: float | None = None) -> Model:
    """Compile as in the notebook: Adam (Keras default lr) + categorical cross-entropy."""
    import keras

    optimizer = (
        keras.optimizers.Adam() if learning_rate is None else keras.optimizers.Adam(learning_rate)
    )
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
