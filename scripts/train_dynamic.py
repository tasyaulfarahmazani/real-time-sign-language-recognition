import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CONFIG
# =========================================================
DATASET_PATH = "dataset/dynamic"

SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 126

MODEL_SAVE_PATH = "models/dynamic_gru.keras"

# =========================================================
# LOAD DATASET
# =========================================================
sequences = []
labels = []

print("Loading dataset...\n")

actions = sorted(os.listdir(DATASET_PATH))

for action in actions:

    action_path = os.path.join(DATASET_PATH, action)

    if not os.path.isdir(action_path):
        continue

    print(f"Loading class : {action}")

    for sequence in os.listdir(action_path):

        sequence_path = os.path.join(
            action_path,
            sequence
        )

        if not os.path.isdir(sequence_path):
            continue

        window = []

        for frame_num in range(SEQUENCE_LENGTH):

            frame_path = os.path.join(
                sequence_path,
                f"{frame_num}.npy"
            )

            if not os.path.exists(frame_path):
                break

            frame = np.load(frame_path)

            window.append(frame)

        if len(window) == SEQUENCE_LENGTH:

            sequences.append(window)
            labels.append(action)

# =========================================================
# CONVERT TO ARRAY
# =========================================================
X = np.array(sequences)

print("\nDataset Loaded")
print("Shape X :", X.shape)

# =========================================================
# LABEL ENCODER
# =========================================================
le = LabelEncoder()

y = le.fit_transform(labels)

y = to_categorical(y)

print("Shape y :", y.shape)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data   :", X_train.shape)
print("Validation Data :", X_val.shape)

# =========================================================
# BUILD MODEL
# =========================================================
model = Sequential()

model.add(
    GRU(
        64,
        return_sequences=True,
        activation="relu",
        input_shape=(
            SEQUENCE_LENGTH,
            FEATURE_LENGTH
        )
    )
)

model.add(Dropout(0.2))

model.add(
    GRU(
        128,
        return_sequences=True,
        activation="relu"
    )
)

model.add(Dropout(0.2))

model.add(
    GRU(
        64,
        return_sequences=False,
        activation="relu"
    )
)

model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(
    Dense(
        len(actions),
        activation="softmax"
    )
)

# =========================================================
# COMPILE MODEL
# =========================================================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL SUMMARY")
model.summary()

# =========================================================
# CALLBACK
# =========================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# =========================================================
# TRAIN MODEL
# =========================================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

# =========================================================
# SAVE MODEL
# =========================================================
os.makedirs("models", exist_ok=True)

model.save(MODEL_SAVE_PATH)

print("\nModel saved:")
print(MODEL_SAVE_PATH)

# =========================================================
# SAVE LABELS
# =========================================================
np.save(
    "models/dynamic_labels.npy",
    le.classes_
)

print("Labels saved")

# =========================================================
# EVALUATION
# =========================================================
loss, accuracy = model.evaluate(
    X_val,
    y_val
)

print(f"\nValidation Accuracy : {accuracy*100:.2f}%")
print(f"Validation Loss     : {loss:.4f}")

# =========================================================
# PREDICTION
# =========================================================
y_pred = model.predict(X_val)

y_true = np.argmax(y_val, axis=1)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=le.classes_
    )
)

# =========================================================
# CREATE RESULTS FOLDER
# =========================================================
os.makedirs(
    "results",
    exist_ok=True
)

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(
    y_true,
    y_pred_classes
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion Matrix saved:"
)
print(
    "results/confusion_matrix.png"
)

# =========================================================
# NORMALIZED CONFUSION MATRIX
# =========================================================
cm_norm = confusion_matrix(
    y_true,
    y_pred_classes,
    normalize="true"
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title(
    "Normalized Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix_normalized.png",
    dpi=300
)

plt.show()

print(
    "Normalized Confusion Matrix saved:"
)
print(
    "results/confusion_matrix_normalized.png"
)

# =========================================================
# TRAINING HISTORY
# =========================================================
plt.figure(figsize=(10,5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_accuracy.png",
    dpi=300
)

plt.show()

# =========================================================
# LOSS HISTORY
# =========================================================
plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_loss.png",
    dpi=300
)

plt.show()

print("\n==============================")
print("ALL RESULTS SAVED")
print("==============================")
print("results/confusion_matrix.png")
print("results/confusion_matrix_normalized.png")
print("results/training_accuracy.png")
print("results/training_loss.png")

# =========================================================
# FINAL ACCURACY COMPARISON GRAPH
# =========================================================

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

labels = [
    "Training",
    "Validation"
]

values = [
    train_accuracy * 100,
    val_accuracy * 100
]


plt.figure(figsize=(8,5))

bars = plt.bar(
    labels,
    values
)

plt.ylim(0,100)

plt.title(
    "Final Model Accuracy Comparison"
)

plt.xlabel(
    "Dataset"
)

plt.ylabel(
    "Accuracy (%)"
)


for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


plt.savefig(
    "results/final_accuracy_comparison.png",
    dpi=300
)


plt.show()


print(
    "Final Accuracy Graph saved:"
)

print(
    "results/final_accuracy_comparison.png"
)