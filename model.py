"""CNN dari nol (tanpa arsitektur/bobot pretrained) untuk klasifikasi daun sehat/sakit.

Input 128x128x3 (nilai 0-1) -> 4 blok [Conv3x3 -> BatchNorm -> ReLU -> MaxPool] (32, 64, 128, 256)
-> GlobalAvgPool -> Dropout 0.4 -> Dense 128 ReLU -> Dense softmax (N kelas).
Bobot diinisialisasi acak (default Keras: Glorot uniform).

Cek cepat:  python model.py
"""
import tensorflow as tf
from tensorflow.keras import layers

from preprocess import CLASS_NAMES, IMG_SIZE

FILTERS = (32, 64, 128, 256)


def build_model(num_classes=len(CLASS_NAMES), img_size=IMG_SIZE, dropout=0.4, bn_momentum=0.99):
    # bn_momentum 0.99 = default Keras (v1); 0.9 = statistik BN lebih cepat mengikuti bobot (val lebih stabil)
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    for i, f in enumerate(FILTERS, 1):
        # use_bias=False: bias redundan karena langsung diikuti BatchNorm
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name="conv{}".format(i))(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name="bn{}".format(i))(x)
        x = layers.ReLU(name="relu{}".format(i))(x)
        x = layers.MaxPooling2D(2, name="pool{}".format(i))(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="leafnet_v1")


def build_transfer(num_classes, img_size=IMG_SIZE, dropout=0.3, weights="imagenet"):
    """Tahap 2b (J2): transfer learning MobileNetV2 (bobot ImageNet; diizinkan dosen untuk transfer learning).
    Input tetap 0-1 (sama dengan leafnet) -> Rescaling ke [-1, 1] sesuai praproses MobileNetV2 di dalam model.
    Return (model, base); base dibekukan untuk fase 1, dibuka sebagian saat fine-tune (train.py)."""
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_mobilenet")(inputs)
    x = base(x, training=False)  # BN base selalu mode inferensi (penting saat fine-tune data kecil)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer"), base


if __name__ == "__main__":
    m = build_model()
    m.summary()
    n = m.count_params()
    assert n < 1_000_000, "parameter {} melebihi target 1 juta".format(n)
    p = m(tf.zeros((2, IMG_SIZE, IMG_SIZE, 3)), training=False).numpy()
    assert p.shape == (2, len(CLASS_NAMES)) and abs(p.sum(axis=1) - 1).max() < 1e-5
    print("OK: {:,} parameter (~{:.1f} MB float32), output {}".format(n, n * 4 / 1e6, p.shape))
