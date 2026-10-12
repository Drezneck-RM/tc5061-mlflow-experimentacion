"""Entrenamiento y evaluación causal de Random Forest para demanda diaria."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DAY = "Day of the week"
TARGET = "Total orders"
FEATURES = [DAY, "lag_1", "mean_past_5", "last_same_weekday"]
WARMUP = 5
PROTOCOL = {
    "task": "next_observed_business_day",
    "row_order": "original_file_order_chronology_unverified",
    "known_covariate": "weekday_of_next_observed_record",
    "warmup_rows": WARMUP,
    "validation_origins_zero_based": list(range(24, 48)),
    "descriptive_test_origins_zero_based": list(range(48, 60)),
    "refit": "each_origin_using_only_preceding_observed_targets",
    "selection_metric": "validation_mae",
    "dummy": "expanding_mean_of_all_preceding_targets",
    "test_status": "previously_viewed_descriptive_only",
    "features": FEATURES,
}


@dataclass(frozen=True)
class ForestConfig:
    n_estimators: int = 100
    max_depth: int = 3
    min_samples_leaf: int = 2
    seed: int = 42

    def __post_init__(self):
        for name in ("n_estimators", "max_depth", "min_samples_leaf"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} debe ser un entero positivo.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed debe estar entre 0 y 2**32 - 1.")

    def to_dict(self):
        return asdict(self)


def validate_data(raw):
    """Conservar el orden original y comprobar los dos campos necesarios."""
    data = raw.loc[:, [DAY, TARGET]].copy().reset_index(drop=True)
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Se requieren valores finitos y sin faltantes.")
    if not data[DAY].isin([2, 3, 4, 5, 6]).all():
        raise ValueError("Los códigos de día válidos son 2, 3, 4, 5 y 6.")
    if (data[TARGET] < 0).any():
        raise ValueError("La demanda no puede ser negativa.")
    return data.astype({DAY: int, TARGET: float})


def load_data(path):
    """Leer la copia local; este experimento fijo requiere 60 registros."""
    raw = pd.read_csv(Path(path))
    validate_data(raw)
    if len(raw) != 60:
        raise ValueError("El protocolo de esta actividad requiere exactamente 60 filas.")
    return raw


def build_features(raw):
    """Cada característica de demanda en t utiliza exclusivamente filas menores a t."""
    data = validate_data(raw)
    past = data[TARGET].shift(1)
    same_weekday = data.groupby(DAY, sort=False)[TARGET].shift(1)
    return pd.DataFrame({
        DAY: data[DAY],
        "lag_1": past,
        "mean_past_5": past.rolling(5, min_periods=5).mean(),
        "last_same_weekday": same_weekday.fillna(past),
    }).loc[:, FEATURES]


def fit_prefix(raw, origin, config):
    """Ajustar el pipeline con filas anteriores al origen y cinco filas de warmup."""
    if not isinstance(origin, (int, np.integer)) or not WARMUP < origin <= len(raw):
        raise ValueError("El origen debe dejar al menos una muestra tras el warmup.")
    history = validate_data(raw.iloc[:int(origin)])
    x_train = build_features(history).iloc[WARMUP:]
    y_train = history[TARGET].iloc[WARMUP:]
    preprocess = ColumnTransformer([
        ("weekday", OneHotEncoder(
            categories=[[2, 3, 4, 5, 6]], handle_unknown="ignore", sparse_output=False,
        ), [DAY]),
        ("history", "passthrough", FEATURES[1:]),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.seed, n_jobs=1,
        )),
    ])
    pipeline.fit(x_train, y_train)
    return pipeline


def rolling_predictions(raw, origins, config):
    """Predecir un registro y actualizar con su resultado antes del siguiente."""
    data = validate_data(raw)
    features = build_features(data)
    rows = []
    for origin in origins:
        origin = int(origin)
        if not WARMUP < origin < len(data):
            raise ValueError("Origen de evaluación fuera del intervalo válido.")
        model = fit_prefix(data, origin, config)
        rows.append({
            "row_1based": origin + 1,
            "train_end_row": origin,
            "training_examples": origin - WARMUP,
            "weekday": int(data[DAY].iloc[origin]),
            "y_true": float(data[TARGET].iloc[origin]),
            "y_pred": float(model.predict(features.iloc[[origin]])[0]),
            "dummy_pred": float(data[TARGET].iloc[:origin].mean()),
        })
    if not rows:
        raise ValueError("Se requiere al menos un origen de evaluación.")
    return pd.DataFrame(rows)


def metric_summary(predictions):
    """MAE/RMSE en unidades originales; sesgo positivo indica sobreestimación."""
    actual = predictions["y_true"].to_numpy(dtype=float)
    predicted = predictions["y_pred"].to_numpy(dtype=float)
    dummy = predictions["dummy_pred"].to_numpy(dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    dummy_mae = float(mean_absolute_error(actual, dummy))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mean_signed_error": float(np.mean(predicted - actual)),
        "dummy_mae": dummy_mae,
        "skill_vs_dummy": float(1 - mae / dummy_mae) if dummy_mae > 0 else None,
        "n_predictions": int(len(actual)),
    }
