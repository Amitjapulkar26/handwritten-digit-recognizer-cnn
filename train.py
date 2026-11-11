"""Train and evaluate a CNN on the MNIST handwritten digit dataset."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "mnist_cnn.keras"
METRICS_PATH = MODEL_DIR / "metrics.json"

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    print("Downloading/loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True
        ),
    ]
    model.fit(
        x_train, y_train,
        epochs=12,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    per_class = {}
    for digit in range(10):
        mask = y_test == digit
        per_class[str(digit)] = float(np.mean(predictions[mask] == y_test[mask]))

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "test_samples": int(len(y_test)),
        "per_digit_accuracy": per_class,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save(MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
