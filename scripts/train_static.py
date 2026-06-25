from html import parser
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import logging
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# ─── Model file untuk MediaPipe Tasks API ─────────────────────────────────────
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def download_model_if_needed(model_path: str = HAND_LANDMARKER_MODEL):
    """Download hand_landmarker.task jika belum ada."""
    if os.path.exists(model_path):
        logging.info(f"Model ditemukan: {model_path}")
        return
    logging.info(f"Mengunduh model MediaPipe ke '{model_path}'...")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)
        logging.info("Download selesai.")
    except Exception as e:
        logging.error(f"Gagal download model: {e}")
        raise


# ─── Feature Extractor (Tasks API) ────────────────────────────────────────────
class HandFeatureExtractor:
    """
    Mengekstrak 126 fitur landmark tangan menggunakan MediaPipe Tasks API
    (kompatibel dengan mediapipe >= 0.10).

    Layout fitur:
      - Tangan Kiri  : indeks   0 – 62  (21 landmark × 3 koordinat)
      - Tangan Kanan : indeks  63 – 125 (21 landmark × 3 koordinat)
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        model_path: str = HAND_LANDMARKER_MODEL,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.model_path = model_path
        self._detector = None  # lazy-init agar aman di thread

    def _init_detector(self):
        if self._detector is None:
            base_options = mp_python.BaseOptions(
                model_asset_path=self.model_path
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract_from_image(self, image_path: str):
        """
        Kembalikan numpy array (126,) atau None jika tidak ada tangan terdeteksi.
        """
        self._init_detector()

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Bungkus ke format MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self._detector.detect(mp_image)

        # Jika tidak ada tangan
        if not results.hand_landmarks or not results.handedness:
            return None

        features = np.zeros(126)

        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            if hand_idx >= self.max_num_hands:
                break

            # Label handedness: 'Left' atau 'Right'
            handedness_label = results.handedness[hand_idx][0].category_name

            # Kumpulkan koordinat 21 landmark
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )  # shape (21, 3)

            # Normalisasi 1: Translation (relatif terhadap pergelangan tangan)
            wrist = landmarks[0].copy()
            landmarks -= wrist

            # Normalisasi 2: Scale (bagi dengan jarak maksimum ke pergelangan)
            max_dist = np.max(np.linalg.norm(landmarks, axis=1))
            if max_dist > 0:
                landmarks /= max_dist

            flat = landmarks.flatten()  # (63,)

            if handedness_label == "Left":
                features[0:63] = flat
            elif handedness_label == "Right":
                features[63:126] = flat

        return features


# ─── Worker function untuk ThreadPoolExecutor ─────────────────────────────────
def process_single_image(args):
    img_path, label, model_path = args
    extractor = HandFeatureExtractor(model_path=model_path)
    features = extractor.extract_from_image(img_path)
    return img_path, label, features


# ─── Ekstraksi fitur seluruh dataset ─────────────────────────────────────────
def extract_dataset_features(
    dataset_dir: str,
    output_csv: str,
    num_workers: int = 4,
    model_path: str = HAND_LANDMARKER_MODEL,
):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        logging.error(f"Direktori dataset tidak ditemukan: {dataset_dir}")
        return None

    # Kumpulkan semua path gambar beserta label kelas
    image_data = []
    classes = sorted(d.name for d in dataset_path.iterdir() if d.is_dir())
    for cls_name in classes:
        cls_dir = dataset_path / cls_name
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
            for img_path in cls_dir.glob(ext):
                image_data.append((str(img_path), cls_name, model_path))

    if not image_data:
        logging.warning(f"Tidak ada gambar ditemukan di {dataset_dir}")
        return None

    logging.info(
        f"Ditemukan {len(image_data)} gambar di {dataset_dir}. "
        f"Memulai ekstraksi dengan {num_workers} worker..."
    )

    results_list = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_image, data): data
            for data in image_data
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Extracting {dataset_path.name}",
        ):
            img_path, label, features = future.result()
            if features is not None:
                results_list.append([label] + features.tolist())
            else:
                failed_count += 1

    logging.info(
        f"Ekstraksi selesai. Berhasil: {len(results_list)}, "
        f"Gagal (tidak ada tangan): {failed_count}"
    )

    if results_list:
        n_features = len(results_list[0]) - 1
        columns = ["label"] + [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(results_list, columns=columns)
        df.to_csv(output_csv, index=False)
        logging.info(f"Fitur disimpan ke {output_csv}")
        return df

    return None


# ─── Training ─────────────────────────────────────────────────────────────────
def train_model(
    train_csv: str,
    val_csv: str,
    model_output_path: str,
    dataset_type: str
):
    logging.info("Memuat data fitur...")
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_val   = val_df.drop("label", axis=1).values
    y_val   = val_df["label"].values

    logging.info(
        f"Training set: {X_train.shape[0]} sampel | "
        f"Validation set: {X_val.shape[0]} sampel"
    )

    # Label encoding (XGBoost butuh integer mulai dari 0)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)

    if XGB_AVAILABLE:
        logging.info("XGBoost tersedia! Training dengan GPU (CUDA)...")
        model = XGBClassifier(
            tree_method="hist",
            device="cuda",
            random_state=42,
            eval_metric="mlogloss",
        )
        param_dist = {
            "n_estimators":     [200, 300, 500, 700],
            "max_depth":        [6, 10, 15, 20],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        n_jobs_search = 1  # Hindari konflik joblib + CUDA
    else:
        logging.info("XGBoost tidak ditemukan. Menggunakan Random Forest CPU...")
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators":    [100, 200, 300, 500],
            "max_depth":       [None, 10, 20, 30, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "bootstrap":       [True, False],
            "criterion":       ["gini", "entropy"],
        }
        n_jobs_search = -1

    logging.info("Memulai RandomizedSearchCV untuk hyperparameter tuning...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=n_jobs_search,
    )
    search.fit(X_train, y_train_enc)

    best_model = search.best_estimator_
    logging.info(f"Hyperparameter terbaik:\n{search.best_params_}")

    # Evaluasi
    logging.info("Mengevaluasi model terbaik pada validation set...")
    y_pred_enc     = best_model.predict(X_val)
    acc            = accuracy_score(y_val_enc, y_pred_enc)
    y_pred_decoded = le.inverse_transform(y_pred_enc)

    logging.info(f"Validation Accuracy: {acc * 100:.2f}%")
    # Grafik Akurasi Training vs Validation
    plot_accuracy_graph(
    search,
    X_train,
    y_train_enc,
    X_val,
    y_val_enc,
    dataset_type
    )
    logging.info(
        "\nClassification Report:\n"
        + classification_report(y_val, y_pred_decoded)
    )

    # Confusion Matrix
    labels = np.unique(y_val)
    cm = confusion_matrix(y_val, y_pred_decoded, labels=labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.title("Confusion Matrix – Validation Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix disimpan ke {cm_path}")

    # Simpan model + LabelEncoder
    os.makedirs(
        os.path.dirname(model_output_path) if os.path.dirname(model_output_path) else ".",
        exist_ok=True,
    )
    joblib.dump({"model": best_model, "label_encoder": le}, model_output_path)
    logging.info(f"Model & LabelEncoder disimpan ke {model_output_path}")
    logging.info("Pipeline training selesai!")


def plot_accuracy_graph(
    search,
    X_train,
    y_train_enc,
    X_val,
    y_val_enc,
    dataset_type
):
    """
    Membuat grafik perbandingan akurasi training dan validation
    dari model terbaik.
    """

    best_model = search.best_estimator_

    train_acc = best_model.score(X_train, y_train_enc)
    val_acc = best_model.score(X_val, y_val_enc)

    labels = ["Training", "Validation"]
    scores = [train_acc * 100, val_acc * 100]

    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, scores)

    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Dataset")
    plt.title("Model Accuracy Comparison")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{score:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    acc_path = f"results/{dataset_type}_accuracy.png"
    plt.savefig(acc_path)
    plt.close()

    logging.info(f"Grafik akurasi disimpan ke {acc_path}")
    

# ─── Entry point ──────────────────────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser(
        description="Hand Sign Recognition – MediaPipe Tasks API + RF/XGBoost"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 1)
    )

    parser.add_argument(
        "--force_extract",
        action="store_true"
    )

    parser.add_argument(
        "--model_file",
        default=HAND_LANDMARKER_MODEL
    )

    args = parser.parse_args()


    # download model sekali
    download_model_if_needed(args.model_file)


    # train dua dataset
    for dataset_type in ["alphabet", "number"]:

        print("\n==============================")
        print(f" TRAINING {dataset_type.upper()} ")
        print("==============================\n")


        train_dir = f"../dataset/static/{dataset_type}/train"
        val_dir = f"../dataset/static/{dataset_type}/val"

        train_csv = f"../features/{dataset_type}_train_features.csv"
        val_csv = f"../features/{dataset_type}_val_features.csv"

        model_out = f"../models/static/{dataset_type}_rf.joblib"


        # ekstraksi train
        if not os.path.exists(train_csv) or args.force_extract:

            extract_dataset_features(
                train_dir,
                train_csv,
                args.workers,
                args.model_file
            )


        # ekstraksi val
        if not os.path.exists(val_csv) or args.force_extract:

            extract_dataset_features(
                val_dir,
                val_csv,
                args.workers,
                args.model_file
            )


        # training
        train_model(
        train_csv,
        val_csv,
        model_out,
        dataset_type
        )


        print(
            f"\nSELESAI TRAIN {dataset_type.upper()}\n"
        )
        
if __name__ == "__main__":
    main()