import os
import json
import math
import time
import shutil
import random
import hashlib
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. ОСНОВНОЙ КОНФИГ
# ============================================================

DATASET_PATH = "dataset.csv"

# Сколько последних столбцов считаем outputs.
# Ты сказал: "снять 1 с самого конца"
OUTPUTS_COUNT = 1

# ----------------------------
# fixed training parameters
# ----------------------------

TASK_TYPE = "regression"
MODEL_TYPE = "feedforward"

LOSS_FUNCTION = "mse"          # "mse", "mae", "huber"
EPOCHS = 200

SEED_MODE = "single"           # "single" или "multi"
RANDOM_SEED = 42
RANDOM_SEEDS = [42, 123, 777]  # используется только если SEED_MODE = "multi"

INPUT_SCALER = "standard"      # пока только "standard"
TARGET_SCALER = "standard"     # "standard" или "none"

DEVICE_POLICY = "cuda_if_available"

CACHE_ENABLED = True
CACHE_DIR = "./cache"

SAVE_BEST_MODEL = True
SAVE_ALL_TRIAL_MODELS = False  # если True, будет сохранять модель каждого trial


# ----------------------------
# algorithm-changed training parameters
# ----------------------------

MAX_LAYERS = 6
MAX_NEURONS_PER_LAYER = 128

ALLOWED_NEURON_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]
ALLOWED_ACTIVATIONS = ["tanh", "relu", "sigmoid"]

LEARNING_RATE_OPTIONS = [1e-4, 3e-4, 1e-3, 3e-3]
BATCH_SIZE_OPTIONS = [16, 32, 64]
OPTIMIZER_OPTIONS = ["adam"]

DROPOUT_RATE_OPTIONS = [0.0]
WEIGHT_DECAY_OPTIONS = [0.0]


# ----------------------------
# fixed evaluation parameters
# ----------------------------

MAIN_METRIC = "val_rmse"
METRIC_DIRECTION = "minimize"

SECONDARY_METRICS = ["val_mae", "val_r2"]

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

EARLY_STOPPING_ENABLED = True
PATIENCE = 20
MIN_DELTA = 0.0001

ARCHITECTURE_PATIENCE = 2
MAX_TRIALS = 100

SEARCH_STRATEGY = "matlab_like_growing"

SEED_SCORE_MODE = "mean"       # "mean", "median", "best"


# ============================================================
# 2. УТИЛИТЫ
# ============================================================

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def file_sha256(path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def stable_json_hash(data: dict) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Для повторяемости. На GPU может немного замедлять.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if DEVICE_POLICY == "cuda_if_available" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_better(new_value: float, best_value: float) -> bool:
    if METRIC_DIRECTION == "minimize":
        return new_value < best_value - MIN_DELTA
    if METRIC_DIRECTION == "maximize":
        return new_value > best_value + MIN_DELTA
    raise ValueError(f"Unknown METRIC_DIRECTION: {METRIC_DIRECTION}")


def aggregate_seed_scores(values: list[float]) -> float:
    if SEED_SCORE_MODE == "mean":
        return float(np.mean(values))
    if SEED_SCORE_MODE == "median":
        return float(np.median(values))
    if SEED_SCORE_MODE == "best":
        if METRIC_DIRECTION == "minimize":
            return float(np.min(values))
        return float(np.max(values))
    raise ValueError(f"Unknown SEED_SCORE_MODE: {SEED_SCORE_MODE}")


# ============================================================
# 3. ЗАГРУЗКА DATASET.CSV
# ============================================================

def looks_like_numeric_header(columns) -> bool:
    """
    Если CSV без заголовков, pandas по умолчанию может принять первую строку
    за названия колонок. Тогда имена колонок будут похожи на числа.
    """
    for col in columns:
        try:
            float(str(col).strip())
        except ValueError:
            return False
    return True


def load_dataset_csv(path: str, outputs_count: int):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл {path} не найден. Положи dataset.csv рядом с main.py"
        )

    # Первая попытка: считаем, что есть header.
    df = pd.read_csv(path)

    # Если названия колонок выглядят как числа, скорее всего header отсутствует.
    # Тогда перечитываем CSV без заголовка.
    if looks_like_numeric_header(df.columns):
        df = pd.read_csv(path, header=None)

    df = df.dropna(axis=0, how="all")

    # Пробуем привести все значения к числам.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"В dataset.csv есть нечисловые или пустые значения в колонках: {bad_cols}. "
            f"Для первой версии нужен полностью числовой CSV."
        )

    data = df.to_numpy(dtype=np.float32)

    if data.ndim != 2:
        raise ValueError("dataset.csv должен быть двумерной таблицей.")

    rows_count, total_columns = data.shape

    if outputs_count <= 0:
        raise ValueError("OUTPUTS_COUNT должен быть >= 1.")

    if outputs_count >= total_columns:
        raise ValueError(
            f"OUTPUTS_COUNT={outputs_count}, но всего столбцов={total_columns}. "
            f"Должен остаться хотя бы 1 input-столбец."
        )

    X = data[:, :-outputs_count]
    y = data[:, -outputs_count:]

    input_size = X.shape[1]
    output_size = y.shape[1]

    print("=" * 70)
    print("DATASET LOADED")
    print("=" * 70)
    print(f"Файл: {path}")
    print(f"Строк: {rows_count}")
    print(f"Всего столбцов: {total_columns}")
    print(f"OUTPUTS_COUNT: {outputs_count}")
    print(f"INPUT_SIZE / количество X-параметров: {input_size}")
    print(f"OUTPUT_SIZE / количество y-параметров: {output_size}")
    print("=" * 70)

    return X, y, input_size, output_size


# ============================================================
# 4. SPLIT + SCALING
# ============================================================

def prepare_data(X: np.ndarray, y: np.ndarray, seed: int):
    """
    Делим:
    train / val / test

    Потом scaler обучаем только на train.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT,
        random_state=seed,
        shuffle=True,
    )

    # validation доля считается от исходного датасета.
    # Поэтому пересчитываем ее относительно train_val.
    val_relative = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=seed,
        shuffle=True,
    )

    x_scaler = None
    y_scaler = None

    if INPUT_SCALER == "standard":
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)
    else:
        raise ValueError(f"Unknown INPUT_SCALER: {INPUT_SCALER}")

    if TARGET_SCALER == "standard":
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)
    elif TARGET_SCALER == "none":
        pass
    else:
        raise ValueError(f"Unknown TARGET_SCALER: {TARGET_SCALER}")

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


# ============================================================
# 5. МОДЕЛЬ
# ============================================================

def make_activation(name: str):
    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    if name == "tanh":
        return nn.Tanh()

    if name == "sigmoid":
        return nn.Sigmoid()

    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)

    if name == "elu":
        return nn.ELU()

    raise ValueError(f"Unknown activation: {name}")


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list[int],
        activation_per_layer: list[str],
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if len(hidden_layers) != len(activation_per_layer):
            raise ValueError(
                "len(hidden_layers) должен быть равен len(activation_per_layer)"
            )

        layers = []
        prev_size = input_size

        for neurons, activation_name in zip(hidden_layers, activation_per_layer):
            layers.append(nn.Linear(prev_size, neurons))
            layers.append(make_activation(activation_name))

            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_size = neurons

        # Для регрессии output activation = linear.
        layers.append(nn.Linear(prev_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_optimizer(name: str, model: nn.Module, learning_rate: float, weight_decay: float):
    name = name.lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def make_loss(name: str):
    name = name.lower()

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "huber":
        return nn.HuberLoss()

    raise ValueError(f"Unknown loss function: {name}")


# ============================================================
# 6. МЕТРИКИ
# ============================================================

def inverse_y_if_needed(y_scaled: np.ndarray, y_scaler):
    if TARGET_SCALER == "standard" and y_scaler is not None:
        return y_scaler.inverse_transform(y_scaled)
    return y_scaled


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true

    mse = float(np.mean(error ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
    }


@torch.no_grad()
def evaluate_model(model, X_np, y_np, y_scaler, device, prefix: str) -> dict:
    model.eval()

    X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)

    pred_scaled = model(X_tensor).detach().cpu().numpy()
    true_scaled = y_np

    pred = inverse_y_if_needed(pred_scaled, y_scaler)
    true = inverse_y_if_needed(true_scaled, y_scaler)

    base = regression_metrics(true, pred)

    return {
        f"{prefix}_mse": base["mse"],
        f"{prefix}_rmse": base["rmse"],
        f"{prefix}_mae": base["mae"],
        f"{prefix}_r2": base["r2"],
    }


# ============================================================
# 7. ОБУЧЕНИЕ ОДНОГО TRIAL
# ============================================================

def train_one_trial(
    trial_config: dict,
    prepared_data: dict,
    input_size: int,
    output_size: int,
    device: torch.device,
    cache_path: Path,
):
    seed = trial_config["seed"]
    set_seed(seed)

    if CACHE_ENABLED and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        print(
            f"[CACHE] {cached['trial_id']} | "
            f"layers={cached['hidden_layers']} | "
            f"act={cached['activation_per_layer']} | "
            f"seed={cached['seed']} | "
            f"val_rmse={cached['metrics']['val_rmse']:.6f}"
        )
        return cached

    hidden_layers = trial_config["hidden_layers"]
    activation_per_layer = trial_config["activation_per_layer"]
    learning_rate = trial_config["learning_rate"]
    batch_size = trial_config["batch_size"]
    optimizer_name = trial_config["optimizer"]
    dropout_rate = trial_config["dropout_rate"]
    weight_decay = trial_config["weight_decay"]

    model = FeedForwardNet(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation_per_layer=activation_per_layer,
        dropout_rate=dropout_rate,
    ).to(device)

    loss_fn = make_loss(LOSS_FUNCTION)
    optimizer = make_optimizer(
        optimizer_name,
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train"]
    X_val = prepared_data["X_val"]
    y_val = prepared_data["y_val"]
    X_test = prepared_data["X_test"]
    y_test = prepared_data["y_test"]
    y_scaler = prepared_data["y_scaler"]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_metric = float("inf") if METRIC_DIRECTION == "minimize" else -float("inf")
    best_state = None
    best_epoch = 0
    early_counter = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_model(
            model=model,
            X_np=X_val,
            y_np=y_val,
            y_scaler=y_scaler,
            device=device,
            prefix="val",
        )

        current_metric = val_metrics[MAIN_METRIC]

        if is_better(current_metric, best_metric):
            best_metric = current_metric
            best_epoch = epoch
            early_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            early_counter += 1

        if EARLY_STOPPING_ENABLED and early_counter >= PATIENCE:
            break

    train_time_sec = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_model(
        model=model,
        X_np=X_train,
        y_np=y_train,
        y_scaler=y_scaler,
        device=device,
        prefix="train",
    )

    val_metrics = evaluate_model(
        model=model,
        X_np=X_val,
        y_np=y_val,
        y_scaler=y_scaler,
        device=device,
        prefix="val",
    )

    test_metrics = evaluate_model(
        model=model,
        X_np=X_test,
        y_np=y_test,
        y_scaler=y_scaler,
        device=device,
        prefix="test",
    )

    metrics = {}
    metrics.update(train_metrics)
    metrics.update(val_metrics)
    metrics.update(test_metrics)

    model_path = None

    if SAVE_ALL_TRIAL_MODELS:
        model_path = str(cache_path.with_suffix(".pt"))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "trial_config": trial_config,
                "input_size": input_size,
                "output_size": output_size,
                "metrics": metrics,
            },
            model_path,
        )

    result = {
        "trial_id": trial_config["trial_id"],
        "config_hash": trial_config["config_hash"],
        "seed": seed,

        "hidden_layers": hidden_layers,
        "activation_per_layer": activation_per_layer,

        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,

        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "train_time_sec": train_time_sec,

        "metrics": metrics,
        "model_path": model_path,
    }

    if CACHE_ENABLED:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    print(
        f"[TRAIN] {result['trial_id']} | "
        f"layers={hidden_layers} | "
        f"act={activation_per_layer} | "
        f"seed={seed} | "
        f"epoch={best_epoch}/{epoch} | "
        f"val_rmse={metrics['val_rmse']:.6f} | "
        f"test_rmse={metrics['test_rmse']:.6f}"
    )

    return result


# ============================================================
# 8. MULTI-SEED ОБЕРТКА
# ============================================================

def train_config_with_seeds(
    base_config: dict,
    prepared_data_by_seed: dict,
    input_size: int,
    output_size: int,
    device: torch.device,
    dataset_hash: str,
    global_trial_index: int,
):
    if SEED_MODE == "multi":
        seeds = RANDOM_SEEDS
    else:
        seeds = [RANDOM_SEED]

    seed_results = []

    for seed in seeds:
        trial_config = dict(base_config)
        trial_config["seed"] = seed
        trial_config["dataset_hash"] = dataset_hash

        config_hash = stable_json_hash(trial_config)
        trial_config["config_hash"] = config_hash
        trial_config["trial_id"] = f"trial_{global_trial_index:05d}_seed_{seed}"

        cache_path = Path(CACHE_DIR) / f"{config_hash}.json"

        result = train_one_trial(
            trial_config=trial_config,
            prepared_data=prepared_data_by_seed[seed],
            input_size=input_size,
            output_size=output_size,
            device=device,
            cache_path=cache_path,
        )

        seed_results.append(result)

    scores = [r["metrics"][MAIN_METRIC] for r in seed_results]
    aggregated_score = aggregate_seed_scores(scores)

    if METRIC_DIRECTION == "minimize":
        best_seed_result = min(seed_results, key=lambda r: r["metrics"][MAIN_METRIC])
    else:
        best_seed_result = max(seed_results, key=lambda r: r["metrics"][MAIN_METRIC])

    return {
        "base_config": base_config,
        "seed_results": seed_results,
        "aggregated_score": aggregated_score,
        "best_seed_result": best_seed_result,
    }


# ============================================================
# 9. ГЕНЕРАЦИЯ КАНДИДАТОВ АРХИТЕКТУРЫ
# ============================================================

def generate_candidates_for_depth(
    depth: int,
    base_layers: list[int] | None,
    base_activations: list[str] | None,
):
    """
    Matlab-like рост:

    depth=1:
        [1], [2], [4], [8] ...

    depth=2:
        берем лучшую прошлую архитектуру, например [10]
        пробуем:
        [10, 1], [10, 2], [10, 4] ...

    depth=3:
        берем лучшую прошлую архитектуру, например [10, 5]
        пробуем:
        [10, 5, 1], [10, 5, 2] ...
    """
    candidates = []

    if depth == 1:
        for n in ALLOWED_NEURON_COUNTS:
            if n <= MAX_NEURONS_PER_LAYER:
                for act in ALLOWED_ACTIVATIONS:
                    candidates.append({
                        "hidden_layers": [n],
                        "activation_per_layer": [act],
                    })
        return candidates

    if base_layers is None or base_activations is None:
        raise ValueError("Для depth > 1 нужны base_layers и base_activations.")

    for n in ALLOWED_NEURON_COUNTS:
        if n <= MAX_NEURONS_PER_LAYER:
            for act in ALLOWED_ACTIVATIONS:
                candidates.append({
                    "hidden_layers": base_layers + [n],
                    "activation_per_layer": base_activations + [act],
                })

    return candidates


def generate_training_variants(architecture_candidate: dict):
    variants = []

    for learning_rate in LEARNING_RATE_OPTIONS:
        for batch_size in BATCH_SIZE_OPTIONS:
            for optimizer in OPTIMIZER_OPTIONS:
                for dropout_rate in DROPOUT_RATE_OPTIONS:
                    for weight_decay in WEIGHT_DECAY_OPTIONS:
                        variants.append({
                            "hidden_layers": architecture_candidate["hidden_layers"],
                            "activation_per_layer": architecture_candidate["activation_per_layer"],

                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "optimizer": optimizer,
                            "dropout_rate": dropout_rate,
                            "weight_decay": weight_decay,

                            "loss_function": LOSS_FUNCTION,
                            "epochs": EPOCHS,
                            "task_type": TASK_TYPE,
                            "model_type": MODEL_TYPE,
                            "main_metric": MAIN_METRIC,
                            "metric_direction": METRIC_DIRECTION,
                        })

    return variants


# ============================================================
# 10. ОСНОВНОЙ ПОИСК
# ============================================================

def run_search():
    ensure_dir(CACHE_DIR)

    dataset_hash = file_sha256(DATASET_PATH)

    X, y, input_size, output_size = load_dataset_csv(
        path=DATASET_PATH,
        outputs_count=OUTPUTS_COUNT,
    )

    device = get_device()

    print(f"DEVICE: {device}")

    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    if SEED_MODE == "multi":
        used_seeds = RANDOM_SEEDS
    else:
        used_seeds = [RANDOM_SEED]

    prepared_data_by_seed = {}

    for seed in used_seeds:
        prepared_data_by_seed[seed] = prepare_data(X, y, seed=seed)

    print("=" * 70)
    print("SEARCH CONFIG")
    print("=" * 70)
    print(f"SEARCH_STRATEGY: {SEARCH_STRATEGY}")
    print(f"MAX_LAYERS: {MAX_LAYERS}")
    print(f"MAX_NEURONS_PER_LAYER: {MAX_NEURONS_PER_LAYER}")
    print(f"ALLOWED_NEURON_COUNTS: {ALLOWED_NEURON_COUNTS}")
    print(f"ALLOWED_ACTIVATIONS: {ALLOWED_ACTIVATIONS}")
    print(f"LEARNING_RATE_OPTIONS: {LEARNING_RATE_OPTIONS}")
    print(f"BATCH_SIZE_OPTIONS: {BATCH_SIZE_OPTIONS}")
    print(f"OPTIMIZER_OPTIONS: {OPTIMIZER_OPTIONS}")
    print(f"MAIN_METRIC: {MAIN_METRIC}")
    print(f"SEED_MODE: {SEED_MODE}")
    print(f"USED_SEEDS: {used_seeds}")
    print("=" * 70)

    global_best_score = float("inf") if METRIC_DIRECTION == "minimize" else -float("inf")
    global_best_result = None

    base_layers = None
    base_activations = None

    architecture_patience_counter = 0
    global_trial_index = 0

    all_group_results = []

    for depth in range(1, MAX_LAYERS + 1):
        print("\n" + "#" * 70)
        print(f"ARCHITECTURE DEPTH: {depth}")
        print("#" * 70)

        candidates = generate_candidates_for_depth(
            depth=depth,
            base_layers=base_layers,
            base_activations=base_activations,
        )

        best_depth_score = float("inf") if METRIC_DIRECTION == "minimize" else -float("inf")
        best_depth_result = None

        for architecture_candidate in candidates:
            training_variants = generate_training_variants(architecture_candidate)

            for base_config in training_variants:
                if global_trial_index >= MAX_TRIALS:
                    print("\nMAX_TRIALS достигнут. Останавливаем поиск.")
                    return finalize_results(global_best_result, all_group_results)

                global_trial_index += 1

                group_result = train_config_with_seeds(
                    base_config=base_config,
                    prepared_data_by_seed=prepared_data_by_seed,
                    input_size=input_size,
                    output_size=output_size,
                    device=device,
                    dataset_hash=dataset_hash,
                    global_trial_index=global_trial_index,
                )

                all_group_results.append(group_result)

                score = group_result["aggregated_score"]

                if is_better(score, best_depth_score):
                    best_depth_score = score
                    best_depth_result = group_result

                if is_better(score, global_best_score):
                    global_best_score = score
                    global_best_result = group_result

                    best_cfg = group_result["base_config"]

                    print("\n>>> NEW GLOBAL BEST")
                    print(f"score: {global_best_score:.6f}")
                    print(f"hidden_layers: {best_cfg['hidden_layers']}")
                    print(f"activation_per_layer: {best_cfg['activation_per_layer']}")
                    print(f"learning_rate: {best_cfg['learning_rate']}")
                    print(f"batch_size: {best_cfg['batch_size']}")
                    print(f"optimizer: {best_cfg['optimizer']}")
                    print("<<<\n")

        if best_depth_result is None:
            print("На этой глубине не найдено результатов.")
            break

        if is_better(best_depth_score, global_best_score) or best_depth_result == global_best_result:
            depth_cfg = best_depth_result["base_config"]

            base_layers = depth_cfg["hidden_layers"]
            base_activations = depth_cfg["activation_per_layer"]

            architecture_patience_counter = 0

            print("\nDEPTH ACCEPTED")
            print(f"base_layers теперь: {base_layers}")
            print(f"base_activations теперь: {base_activations}")
        else:
            architecture_patience_counter += 1

            print("\nDEPTH NOT IMPROVED")
            print(f"architecture_patience_counter: {architecture_patience_counter}")

            if architecture_patience_counter >= ARCHITECTURE_PATIENCE:
                print("ARCHITECTURE_PATIENCE достигнут. Останавливаем рост сети.")
                break

    return finalize_results(global_best_result, all_group_results)


# ============================================================
# 11. СОХРАНЕНИЕ ИТОГОВ
# ============================================================

def finalize_results(global_best_result, all_group_results):
    ensure_dir(CACHE_DIR)

    if global_best_result is None:
        print("Лучший результат не найден.")
        return None

    best_cfg = global_best_result["base_config"]
    best_seed_result = global_best_result["best_seed_result"]

    summary = {
        "best_score": global_best_result["aggregated_score"],
        "main_metric": MAIN_METRIC,
        "metric_direction": METRIC_DIRECTION,
        "best_config": best_cfg,
        "best_seed_result": best_seed_result,
        "seed_results": global_best_result["seed_results"],
        "total_group_trials": len(all_group_results),
    }

    best_json_path = Path(CACHE_DIR) / "best_result.json"

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BEST RESULT")
    print("=" * 70)
    print(f"MAIN_METRIC: {MAIN_METRIC}")
    print(f"BEST SCORE: {summary['best_score']:.6f}")
    print(f"hidden_layers: {best_cfg['hidden_layers']}")
    print(f"activation_per_layer: {best_cfg['activation_per_layer']}")
    print(f"learning_rate: {best_cfg['learning_rate']}")
    print(f"batch_size: {best_cfg['batch_size']}")
    print(f"optimizer: {best_cfg['optimizer']}")
    print(f"dropout_rate: {best_cfg['dropout_rate']}")
    print(f"weight_decay: {best_cfg['weight_decay']}")

    print("\nBEST SEED METRICS:")
    for k, v in best_seed_result["metrics"].items():
        print(f"{k}: {v:.6f}")

    print(f"\nSaved summary: {best_json_path}")
    print("=" * 70)

    return summary


# ============================================================
# 12. ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run_search()