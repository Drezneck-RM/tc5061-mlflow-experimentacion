"""Comparar seis configuraciones fijadas y repetir la base para verificar reproducibilidad."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = [
    ("base", {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 2, "seed": 42}),
    ("trees_300", {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 2, "seed": 42}),
    ("depth_1", {"n_estimators": 100, "max_depth": 1, "min_samples_leaf": 2, "seed": 42}),
    ("depth_6", {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 2, "seed": 42}),
    ("leaf_1", {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 1, "seed": 42}),
    ("leaf_4", {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 4, "seed": 42}),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ejecutar seis configuraciones, variando un hiperparámetro a la vez, y repetir la base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="Mostrar esta ayuda y salir.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Carpeta nueva; por defecto results/comparison-<fecha UTC>.")
    parser.add_argument("--tracking-uri", default=None,
                        help="URI MLflow opcional; si se omite usa el SQLite del proyecto.")
    parser.add_argument("--experiment-name", default="TC5061_Experimentacion_CLI_Demanda",
                        help="Experimento compartido por todas las ejecuciones.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "daily_demand.csv",
                        help="CSV local de 60 registros.")
    parser.add_argument("--evaluate-best-test", action="store_true",
                        help="Después de seleccionar por validación, registrar una ejecución descriptiva adicional del elegido.")
    return parser.parse_args(argv)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_configuration(args, output_dir, name, params, evaluate_test=False):
    destination = output_dir / name
    command = [sys.executable, str(ROOT / "train.py"),
               "--data-path", str(args.data_path.resolve()),
               "--experiment-name", args.experiment_name,
               "--run-name", name, "--output-dir", str(destination)]
    for key, value in params.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if args.tracking_uri:
        command.extend(["--tracking-uri", args.tracking_uri])
    if evaluate_test:
        command.append("--evaluate-test")
    print(f"Ejecutando {name}: {params}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads((destination / "summary.json").read_text(encoding="utf-8"))


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or ROOT / "results" / f"comparison-{stamp}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Usa una carpeta vacía para conservar resultados anteriores: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Guardar el diseño antes de obtener cualquier resultado.
    write_json(output_dir / "frozen_design.json", {
        "configurations": [{"name": name, **params} for name, params in CONFIGURATIONS],
        "selection": "minimum_validation_mae_ties_follow_list_order",
        "reproducibility_run": "base_repeat",
        "changing_one_parameter_at_a_time": True,
        "interactions_evaluated": False,
        "test_used_for_selection": False,
    })
    summaries = [(name, run_configuration(args, output_dir, name, params))
                 for name, params in CONFIGURATIONS]
    base = summaries[0][1]
    repeat = run_configuration(args, output_dir, "base_repeat", CONFIGURATIONS[0][1])
    original_predictions = pd.read_csv(output_dir / "base" / "validation_predictions.csv")
    repeated_predictions = pd.read_csv(output_dir / "base_repeat" / "validation_predictions.csv")
    keys = sorted(base["metrics"])
    checks = {
        "params_equal": base["params"] == repeat["params"],
        "hashes_equal": base["hashes"] == repeat["hashes"],
        "metric_keys_equal": keys == sorted(repeat["metrics"]),
        "metrics_equal": bool(np.allclose(
            [base["metrics"][key] for key in keys],
            [repeat["metrics"][key] for key in keys], rtol=1e-12, atol=1e-12,
        )),
        "prediction_origins_equal": original_predictions["row_1based"].equals(repeated_predictions["row_1based"]),
        "predictions_equal": bool(np.allclose(
            original_predictions[["y_true", "y_pred", "dummy_pred"]].to_numpy(),
            repeated_predictions[["y_true", "y_pred", "dummy_pred"]].to_numpy(),
            rtol=1e-12, atol=1e-12,
        )),
    }
    write_json(output_dir / "reproducibility.json", {
        "base_run_id": base["run_id"], "repeat_run_id": repeat["run_id"],
        "checks": checks, "all_passed": all(checks.values()),
        "ignored_run_specific_fields": ["run_id", "model_uri", "run_name"],
        "numeric_tolerance": {"rtol": 1e-12, "atol": 1e-12},
    })
    if not all(checks.values()):
        raise AssertionError(f"La reproducción no coincide: {checks}")
    rows = [{"configuration": name, "run_id": summary["run_id"],
             **summary["params"], **summary["metrics"], "is_repeat": False}
            for name, summary in summaries]
    rows.append({"configuration": "base_repeat", "run_id": repeat["run_id"],
                 **repeat["params"], **repeat["metrics"], "is_repeat": True})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    selected_name, selected = min(summaries, key=lambda item: item[1]["metrics"]["validation_mae"])
    sensitivity = []
    for name, summary in summaries[1:]:
        changed = [key for key in base["params"] if summary["params"][key] != base["params"][key]]
        assert len(changed) == 1
        parameter = changed[0]
        sensitivity.append({
            "configuration": name, "parameter": parameter,
            "base_value": base["params"][parameter], "changed_value": summary["params"][parameter],
            "base_validation_mae": base["metrics"]["validation_mae"],
            "changed_validation_mae": summary["metrics"]["validation_mae"],
            "mae_difference_changed_minus_base": summary["metrics"]["validation_mae"] - base["metrics"]["validation_mae"],
            "rmse_difference_changed_minus_base": summary["metrics"]["validation_rmse"] - base["metrics"]["validation_rmse"],
        })
    pd.DataFrame(sensitivity).to_csv(output_dir / "paired_sensitivity.csv", index=False)
    result = {"experiment_name": args.experiment_name,
              "selected_configuration": selected_name, "selected_run_id": selected["run_id"],
              "selected_params": selected["params"], "selection_metric": "validation_mae",
              "selection_score": selected["metrics"]["validation_mae"],
              "reproducibility_passed": True, "configurations": rows,
              "interpretation": "Exploratory selection on shared validation rows; not an unbiased final performance estimate."}
    write_json(output_dir / "comparison.json", result)
    from mlflow.tracking import MlflowClient
    tracking_uri = args.tracking_uri or "sqlite:///" + (ROOT / "mlruns.sqlite").as_posix()
    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_artifact(base["run_id"], str(output_dir / "reproducibility.json"), "comparison")
    for filename in ["comparison.csv", "comparison.json", "paired_sensitivity.csv", "frozen_design.json"]:
        client.log_artifact(selected["run_id"], str(output_dir / filename), "comparison")
    for name, summary in summaries:
        client.set_tag(summary["run_id"], "comparison_group", output_dir.name)
        client.set_tag(summary["run_id"], "configuration", name)
        client.set_tag(summary["run_id"], "selected_by_validation", str(name == selected_name).lower())
        client.set_tag(summary["run_id"], "is_reproducibility_repeat", "false")
    client.set_tag(repeat["run_id"], "comparison_group", output_dir.name)
    client.set_tag(repeat["run_id"], "configuration", "base_repeat")
    client.set_tag(repeat["run_id"], "is_reproducibility_repeat", "true")
    client.set_tag(repeat["run_id"], "selected_by_validation", "false")
    # La selección ya quedó persistida; esta evaluación nunca cambia el ganador.
    if args.evaluate_best_test:
        descriptive = run_configuration(args, output_dir, f"{selected_name}_descriptive_test",
                                        selected["params"], evaluate_test=True)
        write_json(output_dir / "selected_descriptive_test.json", descriptive)
    print(comparison.to_string(index=False))
    print(f"Seleccionado por MAE de validación: {selected_name}")
    print(f"Comparación y verificación guardadas en: {output_dir}")
    return result


if __name__ == "__main__":
    main()
