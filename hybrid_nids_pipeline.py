"""
Hybrid Network Intrusion Detection with Temporal Flow Windows + XAI

Architecture: [Flow Windows] -> Dense -> Bi-LSTM -> Multi-Head Attention
              (+ Residual + LayerNorm) -> XGBoost

Key contributions:
  1. Temporal Flow Windowing — groups consecutive flows into windows,
     giving the Bi-LSTM real sequences instead of trivial single-step input.
  2. Focal Loss — addresses CIC-IDS-2017's heavy class imbalance by
     down-weighting easy examples during DL pre-training.
  3. Dual-Layer Explainability — SHAP on XGBoost features + gradient
     attribution back to original network features.

Dataset: CIC-IDS-2017 (Monday, Wednesday, Friday)
"""

from dataclasses import dataclass
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, roc_curve, auc,
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM,
    MultiHeadAttention, GlobalAveragePooling1D,
    Dropout, Add, LayerNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All pipeline hyperparameters in one place."""

    sample_per_file: int = 60_000
    window_size: int = 10          # consecutive flows per temporal window
    test_size: float = 0.2
    seed: int = 42

    # DL architecture
    dense_units: int = 64
    lstm_units: int = 64
    attn_heads: int = 4
    attn_key_dim: int = 32
    dropout: float = 0.3

    # focal loss params (Lin et al., ICCV 2017)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # training
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 3

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.1

    # SHAP
    shap_n_samples: int = 500

    # paths (auto-set)
    data_dir: str = ""
    model_dir: str = ""
    output_dir: str = ""

    def __post_init__(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base, "data")
        self.model_dir = os.path.join(base, "models")
        self.output_dir = os.path.join(base, "outputs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_and_sample(path: str, n: int, seed: int) -> pd.DataFrame:
    """Load CSV with stratified sampling to preserve class ratios."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if len(df) <= n:
        return df

    try:
        if df["Label"].value_counts().min() < 2:
            raise ValueError
        _, sampled = train_test_split(
            df, test_size=n, random_state=seed, stratify=df["Label"]
        )
        return sampled.reset_index(drop=True)
    except ValueError:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, drop incomplete rows."""
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Cleaned: removed {before - len(df)} bad rows, {len(df)} remaining")
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame):
    """Encode string labels -> integers. Returns X, y, encoder, feature_names."""
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].values.astype(np.float32)
    return X, y, le, feature_cols


def create_flow_windows(X, y, window_size):
    """
    Group consecutive flows into non-overlapping temporal windows.

    This is the core novelty — instead of treating each flow independently
    (timesteps=1, which makes LSTM/Attention trivial), we group W consecutive
    flows so the Bi-LSTM can learn temporal attack patterns.

    Label per window = majority vote among constituent flows.

    Returns:
        X_windows: (n_windows, window_size, n_features)
        y_windows: (n_windows,) majority-vote labels
    """
    n_windows = len(X) // window_size
    n_used = n_windows * window_size

    X_windows = X[:n_used].reshape(n_windows, window_size, -1)
    y_grouped = y[:n_used].reshape(n_windows, window_size)

    # majority vote: label = most common class in the window
    y_windows = np.array([np.bincount(row).argmax() for row in y_grouped])

    return X_windows, y_windows


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for class imbalance — down-weights easy/majority samples
    so the model focuses on hard/minority attack classes.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    @tf.function
    def _focal(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_oh = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        ce = -y_oh * tf.math.log(y_pred)
        w = alpha * y_oh * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(w * ce, axis=-1))
    return _focal


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def build_feature_extractor(n_features, window_size, cfg: Config):
    """
    DL feature extractor with residual attention block.

    Input(W, F) -> Dense(64) -> Dropout
                -> Bi-LSTM(64) returns sequences -> (W, 128)
                -> MultiHeadAttention(4 heads)
                -> [+ Residual skip from Bi-LSTM output] -> LayerNorm
                -> GlobalAveragePooling1D -> 128-dim feature vector
    """
    inp = Input(shape=(window_size, n_features), name="flow_window")

    x = Dense(cfg.dense_units, activation="relu", name="dense_compress")(inp)
    x = Dropout(cfg.dropout)(x)

    x = Bidirectional(
        LSTM(cfg.lstm_units, return_sequences=True),
        name="bi_lstm"
    )(x)

    # self-attention with residual connection for gradient flow
    attn = MultiHeadAttention(
        num_heads=cfg.attn_heads, key_dim=cfg.attn_key_dim, name="mha"
    )(x, x)
    x = Add(name="residual")([x, attn])
    x = LayerNormalization(name="layer_norm")(x)

    features = GlobalAveragePooling1D(name="pool")(x)

    return Model(inp, features, name="feature_extractor")


def build_trainable_model(extractor, n_classes, cfg):
    """Attach softmax classifier head for DL pre-training phase."""
    out = Dense(n_classes, activation="softmax", name="classifier")(extractor.output)
    model = Model(extractor.input, out, name="dl_trainer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=focal_loss(cfg.focal_gamma, cfg.focal_alpha),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation & Plotting
# ---------------------------------------------------------------------------

def full_evaluation(y_true, y_pred, y_prob, class_names, output_dir):
    """Compute all metrics, save report + plots for the paper."""
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names)

    # text report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy:      {acc:.4f}\n")
        f.write(f"F1 (macro):    {f1_m:.4f}\n")
        f.write(f"F1 (weighted): {f1_w:.4f}\n\n")
        f.write(report)

    print(f"\n  Accuracy:      {acc*100:.2f}%")
    print(f"  F1 (macro):    {f1_m:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    print(f"\n{report}")

    _plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    if y_prob is not None:
        _plot_roc_curves(y_true, y_prob, class_names, output_dir)

    return {"accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w}


def _plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: confusion_matrix.png")


def _plot_roc_curves(y_true, y_prob, class_names, output_dir):
    n_cls = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_cls))
    colors = plt.cm.Set2(np.linspace(0, 1, n_cls))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, c) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: roc_curves.png")


def plot_training_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Train", lw=2)
    ax1.plot(history.history["val_loss"], label="Val", lw=2)
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Train", lw=2)
    ax2.plot(history.history["val_accuracy"], label="Val", lw=2)
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=200)
    plt.close(fig)
    print("  Saved: training_history.png")


# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------

def run_ablation(X_train_w, X_test_w, y_train, y_test,
                 extractor, full_model, class_names, cfg):
    """
    Three-way comparison on the same windowed data:
      1. XGBoost on flattened windows (no learned features)
      2. DL-only (softmax head, no XGBoost)
      3. Hybrid DL features -> XGBoost (proposed method)

    Returns dict of results, the trained hybrid XGBoost, and its predictions.
    """
    results = {}

    # 1) XGBoost baseline — flatten windows to (n, W*F)
    print("\n  [Ablation 1/3] XGBoost on flattened windows...")
    X_tr_flat = X_train_w.reshape(len(X_train_w), -1)
    X_te_flat = X_test_w.reshape(len(X_test_w), -1)

    xgb_base = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators, max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate, eval_metric="mlogloss",
        random_state=cfg.seed, n_jobs=-1,
    )
    xgb_base.fit(X_tr_flat, y_train)
    yp = xgb_base.predict(X_te_flat)
    results["XGBoost (flat)"] = {
        "accuracy": accuracy_score(y_test, yp),
        "f1_macro": f1_score(y_test, yp, average="macro"),
        "f1_weighted": f1_score(y_test, yp, average="weighted"),
    }

    # 2) DL-only — the trained model with softmax head
    print("  [Ablation 2/3] DL-only model...")
    dl_probs = full_model.predict(X_test_w, batch_size=cfg.batch_size, verbose=0)
    yp_dl = np.argmax(dl_probs, axis=-1)
    results["DL Only"] = {
        "accuracy": accuracy_score(y_test, yp_dl),
        "f1_macro": f1_score(y_test, yp_dl, average="macro"),
        "f1_weighted": f1_score(y_test, yp_dl, average="weighted"),
    }

    # 3) Hybrid — DL features -> XGBoost (our method)
    print("  [Ablation 3/3] Hybrid: DL features -> XGBoost...")
    X_tr_feat = extractor.predict(X_train_w, batch_size=cfg.batch_size, verbose=0)
    X_te_feat = extractor.predict(X_test_w, batch_size=cfg.batch_size, verbose=0)

    xgb_hybrid = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators, max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate, eval_metric="mlogloss",
        random_state=cfg.seed, n_jobs=-1,
    )
    xgb_hybrid.fit(X_tr_feat, y_train)
    yp_hybrid = xgb_hybrid.predict(X_te_feat)
    results["Hybrid (Ours)"] = {
        "accuracy": accuracy_score(y_test, yp_hybrid),
        "f1_macro": f1_score(y_test, yp_hybrid, average="macro"),
        "f1_weighted": f1_score(y_test, yp_hybrid, average="weighted"),
    }

    return results, xgb_hybrid, X_te_feat, yp_hybrid


def plot_ablation(results, output_dir):
    methods = list(results.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    labels = ["Accuracy", "F1 (Macro)", "F1 (Weighted)"]
    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [results[k][m] for k in methods]
        bars = ax.bar(x + i * w, vals, w, label=lab)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study", fontweight="bold")
    ax.set_xticks(x + w)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_study.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ablation_study.png")


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def shap_analysis(xgb_model, X_feat, output_dir, n=500):
    """SHAP TreeExplainer on XGBoost over deep features."""
    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed — pip install shap")
        return

    idx = np.random.choice(len(X_feat), min(n, len(X_feat)), replace=False)
    X_s = X_feat[idx]
    names = [f"deep_feat_{i}" for i in range(X_s.shape[1])]

    print(f"  Running SHAP TreeExplainer on {len(X_s)} samples...")
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(X_s)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_s, feature_names=names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP: Deep Feature Importance (XGBoost Layer)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_deep_features.png"),
                dpi=200, bbox_inches="tight")
    plt.close("all")
    print("  Saved: shap_deep_features.png")


def gradient_attribution(extractor, X_sample, feature_names, output_dir):
    """
    Gradient-based attribution: map DL feature importance back to
    original network features (packet length, flow duration, etc.).

    Computes mean |gradient| of extractor output w.r.t. input features,
    averaged across a sample of test windows.
    """
    subset = X_sample[:200]  # cap for speed
    X_t = tf.cast(subset, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_t)
        feats = extractor(X_t, training=False)
        target = tf.reduce_sum(feats)

    grads = tape.gradient(target, X_t)
    importance = tf.reduce_mean(tf.abs(grads), axis=[0, 1]).numpy()

    # top-20 original features
    top_k = min(20, len(feature_names))
    top_idx = np.argsort(importance)[-top_k:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("viridis", top_k)
    ax.barh(range(top_k), top_vals, color=palette)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Mean |Gradient|")
    ax.set_title("Gradient Attribution: Original Feature Importance (DL Layer)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_attribution.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: gradient_attribution.png")


# ---------------------------------------------------------------------------
# Single-Sample Inference
# ---------------------------------------------------------------------------

def predict_sample(raw_features, models_dir=None):
    """
    Run the full pipeline on a single raw flow for live inference.

    In production, you'd buffer W consecutive flows to form a real window.
    For demo purposes, the single flow is replicated across the window.
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    extractor = tf.keras.models.load_model(
        os.path.join(models_dir, "feature_extractor.keras"),
        compile=False,
    )
    xgb = joblib.load(os.path.join(models_dir, "xgb_classifier.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    win_sz = joblib.load(os.path.join(models_dir, "window_size.joblib"))

    raw = np.array(raw_features, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(raw)
    # replicate flow to fill window (single-flow demo mode)
    windowed = np.tile(scaled, (win_sz, 1)).reshape(1, win_sz, -1)

    deep = extractor.predict(windowed, verbose=0)
    pred = xgb.predict(deep)[0]
    proba = xgb.predict_proba(deep)[0]
    label = le.inverse_transform([pred])[0]

    return {
        "label": label,
        "confidence": float(np.max(proba)),
        "probabilities": dict(zip(le.classes_.tolist(), proba.tolist())),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    cfg = Config()

    csv_files = {
        "Monday":    os.path.join(cfg.data_dir, "Monday-WorkingHours.pcap_ISCX.csv"),
        "Wednesday": os.path.join(cfg.data_dir, "Wednesday-workingHours.pcap_ISCX.csv"),
        "Friday":    os.path.join(cfg.data_dir,
                                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    }

    print("=" * 60)
    print("HYBRID NIDS — Temporal Flow Windows + XAI")
    print("=" * 60)

    # ---- 1. Load & merge multi-day data ----
    frames = []
    for day, path in csv_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        print(f"\n  Loading {day}...", end=" ")
        df = load_and_sample(path, cfg.sample_per_file, cfg.seed)
        print(f"{len(df):,} rows")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"\n  Combined: {len(df):,} rows")
    df = clean(df)

    # class distribution plot
    counts = df["Label"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Class Distribution (CIC-IDS-2017)", fontweight="bold")
    ax.set_xlabel("Traffic Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "class_distribution.png"), dpi=200)
    plt.close(fig)
    print("  Saved: class_distribution.png")

    # ---- 2. Encode labels ----
    X_raw, y_raw, le, feat_names = encode_labels(df)
    del df

    print(f"\n  Features: {len(feat_names)}, Classes: {len(le.classes_)}")
    for i, name in enumerate(le.classes_):
        print(f"    {i}: {name} ({np.sum(y_raw == i):,})")

    # drop classes with < 2 samples (can't stratify otherwise)
    unique_cls, cls_counts = np.unique(y_raw, return_counts=True)
    rare = unique_cls[cls_counts < 2]
    if len(rare) > 0:
        print(f"  Dropping {len(rare)} class(es) with <2 samples")
        mask = ~np.isin(y_raw, rare)
        X_raw = X_raw[mask]
        text_labels = le.inverse_transform(y_raw[mask])
        le = LabelEncoder()
        y_raw = le.fit_transform(text_labels)
        print(f"  Remaining: {len(le.classes_)} classes, {len(y_raw):,} samples")

    # ---- 3. Temporal flow windows ----
    print(f"\n  Creating temporal windows (size={cfg.window_size})...")
    X_win, y_win = create_flow_windows(X_raw, y_raw, cfg.window_size)
    del X_raw, y_raw
    print(f"  -> {X_win.shape[0]:,} windows of {cfg.window_size} flows "
          f"x {X_win.shape[2]} features")

    # Map back to text to drop any window-level rare classes and re-encode
    # This prevents XGBoost crashing if a class is entirely lost after majority voting
    unique_win_cls, win_cls_counts = np.unique(y_win, return_counts=True)
    rare_win = unique_win_cls[win_cls_counts < 2]
    
    mask = ~np.isin(y_win, rare_win)
    if not np.all(mask):
        print(f"  Dropping {np.sum(~mask)} windows belonging to classes with <2 samples.")
        X_win = X_win[mask]
        y_win = y_win[mask]
        
    text_labels = le.inverse_transform(y_win)
    le = LabelEncoder()
    y_win = le.fit_transform(text_labels)
    print(f"  Classes remaining after windowing: {len(le.classes_)} ({le.classes_.tolist()})")

    # ---- 4. Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_win, y_win,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y_win,
    )
    del X_win, y_win
    print(f"  Train: {len(y_train):,} | Test: {len(y_test):,} windows")

    # ---- 5. Feature scaling (fit on train only — no leakage) ----
    scaler = StandardScaler()
    n_tr, W, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(n_tr, W, F).astype(np.float32)
    n_te = len(X_test)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(n_te, W, F).astype(np.float32)
    print("  Scaling complete (fitted on train flows only)")

    # ---- 6. Build & train DL feature extractor ----
    print("\n" + "=" * 60)
    print("DL Feature Extractor")
    print("  Dense -> Bi-LSTM -> Multi-Head Attention (+ Residual)")
    print("=" * 60)

    extractor = build_feature_extractor(F, W, cfg)
    extractor.summary()

    n_classes = len(le.classes_)
    full_model = build_trainable_model(extractor, n_classes, cfg)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    print(f"\n  Training for up to {cfg.epochs} epochs (focal loss)...")
    history = full_model.fit(
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    plot_training_history(history, cfg.output_dir)

    # ---- 7. Ablation study ----
    print("\n" + "=" * 60)
    print("Ablation Study")
    print("=" * 60)

    abl_results, xgb_model, X_test_feat, y_pred = run_ablation(
        X_train, X_test, y_train, y_test,
        extractor, full_model, le.classes_, cfg,
    )

    print("\n  Results:")
    print(f"  {'Method':<22} {'Acc':>8} {'F1-Mac':>8} {'F1-Wgt':>8}")
    print("  " + "-" * 48)
    for method, s in abl_results.items():
        print(f"  {method:<22} {s['accuracy']:>8.4f} {s['f1_macro']:>8.4f} "
              f"{s['f1_weighted']:>8.4f}")

    plot_ablation(abl_results, cfg.output_dir)
    pd.DataFrame(abl_results).T.to_csv(
        os.path.join(cfg.output_dir, "ablation_results.csv"))

    # ---- 8. Full evaluation of hybrid model ----
    print("\n" + "=" * 60)
    print("Final Evaluation — Hybrid Model")
    print("=" * 60)

    y_prob = xgb_model.predict_proba(X_test_feat)
    full_evaluation(y_test, y_pred, y_prob, le.classes_, cfg.output_dir)

    # ---- 9. Explainability ----
    print("\n" + "=" * 60)
    print("Explainability Analysis")
    print("=" * 60)

    shap_analysis(xgb_model, X_test_feat, cfg.output_dir, cfg.shap_n_samples)
    gradient_attribution(extractor, X_test, feat_names, cfg.output_dir)

    # ---- 10. Save all artifacts ----
    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)

    extractor.save(os.path.join(cfg.model_dir, "feature_extractor.keras"))
    joblib.dump(xgb_model, os.path.join(cfg.model_dir, "xgb_classifier.joblib"))
    joblib.dump(le, os.path.join(cfg.model_dir, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(cfg.model_dir, "scaler.joblib"))
    joblib.dump(cfg.window_size, os.path.join(cfg.model_dir, "window_size.joblib"))

    for name in ["feature_extractor.keras", "xgb_classifier.joblib",
                  "label_encoder.joblib", "scaler.joblib", "window_size.joblib"]:
        print(f"  -> {name}")

    # ---- Done ----
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    acc = abl_results["Hybrid (Ours)"]["accuracy"]
    print(f"  Hybrid Accuracy: {acc*100:.2f}%")
    print(f"  Models saved to: {cfg.model_dir}/")
    print(f"  Plots saved to:  {cfg.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
