"""Ejecutar una configuración reproducible y registrarla en MLflow."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demand_forecasting import (  # noqa: E402
    FEATURES, PROTOCOL, ForestConfig, build_features, fit_prefix,
    load_data, metric_summary, rolling_predictions,
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo.")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("Debe estar entre 0 y 2**32 - 1.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pronosticar la siguiente observación de demanda y registrar el experimento.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="Mostrar esta ayuda y salir.")
    parser.add_argument("--n-estimators", type=positive_int, default=100,
                        help="Número de árboles del Random Forest.")
    parser.add_argument("--max-depth", type=positive_int, default=3,
                        help="Profundidad máxima de cada árbol.")
    parser.add_argument("--min-samples-leaf", type=positive_int, default=2,
                        help="Muestras mínimas por hoja.")
    parser.add_argument("--seed", type=seed_value, default=42,
                        help="Semilla para Python, NumPy y Random Forest.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "daily_demand.csv",
                        help="CSV local del dataset UCI 409 con sus 60 filas en orden original.")
    parser.add_argument("--tracking-uri", default="sqlite:///" + (ROOT / "mlruns.sqlite").as_posix(),
                        help="URI de seguimiento MLflow; por defecto, SQLite dentro del proyecto.")
    parser.add_argument("--experiment-name", default="TC5061_Experimentacion_CLI_Demanda",
                        help="Nombre del experimento MLflow.")
    parser.add_argument("--run-name", default="random_forest_cli",
                        help="Nombre visible de esta ejecución.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Carpeta de resultados; si se omite se usa results/<run_id> del proyecto.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluar también las 12 filas finales, ya vistas; solo análisis descriptivo.")
    return parser.parse_args(argv)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def source_manifest():
    paths = [ROOT / "train.py", ROOT / "requirements.txt"]
    paths.extend(sorted((ROOT / "src" / "demand_forecasting").glob("*.py")))
    files = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    digest = hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()
    return files, digest


def save_plot(predictions, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, label, color, style in [
        ("y_true", "Demanda observada", "black", "-"),
        ("y_pred", "Random Forest", "#1764A0", "-"),
        ("dummy_pred", "Dummy: media acumulada", "#B45220", "--"),
    ]:
        ax.plot(predictions["row_1based"], predictions[column],
                label=label, color=color, linestyle=style, marker=".")
    ax.set(xlabel="Fila original (orden temporal no confirmado)",
           ylabel="Demanda: unidades originales", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from importlib.metadata import version

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    config = ForestConfig(args.n_estimators, args.max_depth, args.min_samples_leaf, args.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    data_path = args.data_path.resolve()
    raw = load_data(data_path)
    source_files, source_hash = source_manifest()
    protocol_hash = hashlib.sha256(canonical_json(PROTOCOL).encode("utf-8")).hexdigest()
    data_hash = sha256_file(data_path)
    pinned_packages = [line.split("==")[0] for line in
                       (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                       if line and not line.startswith("#")]
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: version(name) for name in pinned_packages}}
    hashes = {"dataset_sha256": data_hash, "model_source_sha256": source_hash,
              "protocol_sha256": protocol_hash}
    artifact_root = ROOT / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(args.experiment_name, artifact_location=artifact_root.as_uri())
    experiment = mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        output_dir = (args.output_dir or ROOT / "results" / run_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "summary.json").exists():
            raise FileExistsError(f"La carpeta ya contiene resultados: {output_dir}")
        mlflow.log_params({**config.to_dict(), **hashes,
                           "model_class": "RandomForestRegressor", "n_jobs": 1,
                           "validation_start_row": 25, "validation_end_row": 48,
                           "refit": "daily_expanding", "warmup_rows": 5,
                           "features": ", ".join(FEATURES),
                           "evaluate_test": args.evaluate_test})
        mlflow.set_tags({"protocol": "next_observed_business_day",
                         "chronology": "original_order_unverified",
                         "test_role": "previously_viewed_descriptive",
                         "run_role": "cli_configuration"})
        validation = rolling_predictions(raw, PROTOCOL["validation_origins_zero_based"], config)
        validation_metrics = metric_summary(validation)
        metrics = {f"validation_{key}": value for key, value in validation_metrics.items()}
        validation.to_csv(output_dir / "validation_predictions.csv", index=False)
        save_plot(validation, output_dir / "validation_predictions.png", "Validación: predicción de un registro adelante")
        if args.evaluate_test:
            test = rolling_predictions(raw, PROTOCOL["descriptive_test_origins_zero_based"], config)
            metrics.update({f"descriptive_test_{key}": value for key, value in metric_summary(test).items()})
            test.to_csv(output_dir / "descriptive_test_predictions.csv", index=False)
            save_plot(test, output_dir / "descriptive_test_predictions.png", "Prueba previamente observada: comparación descriptiva")
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)
        (output_dir / "metrics.json").write_text(canonical_json(metrics), encoding="utf-8")
        (output_dir / "protocol.json").write_text(canonical_json(PROTOCOL), encoding="utf-8")
        (output_dir / "source_manifest.json").write_text(canonical_json(source_files), encoding="utf-8")
        (output_dir / "environment.json").write_text(canonical_json(environment), encoding="utf-8")

        # Este estado se ajusta con filas 1..48; la fila 49 solo aporta variables conocidas.
        model = fit_prefix(raw, 48, config)
        input_example = build_features(raw).iloc[[48]].reset_index(drop=True)
        expected_prediction = model.predict(input_example)
        info = mlflow.sklearn.log_model(
            sk_model=model, name="random_forest_origin48",
            signature=infer_signature(input_example, expected_prediction),
            input_example=input_example,
            pip_requirements=str(ROOT / "requirements.txt"),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        reloaded = mlflow.pyfunc.load_model(info.model_uri)
        recovered_prediction = reloaded.predict(input_example)
        np.testing.assert_allclose(expected_prediction, recovered_prediction, rtol=0, atol=1e-12)
        verification = {"model_reloaded": True, "prediction_equal": True,
                        "snapshot_train_end_row": 48,
                        "prediction_row": 49,
                        "prediction": float(expected_prediction[0]),
                        "target_used_in_verification": False}
        (output_dir / "reload_verification.json").write_text(canonical_json(verification), encoding="utf-8")
        summary = {"run_id": run_id, "experiment_id": experiment.experiment_id,
                   "experiment_name": args.experiment_name, "run_name": args.run_name,
                   "params": config.to_dict(), "hashes": hashes, "metrics": metrics,
                   "model_uri": info.model_uri, "model_reloaded": True,
                   "evaluate_test": args.evaluate_test,
                   "snapshot_train_end_row": 48, "origin48_prediction": float(expected_prediction[0])}
        (output_dir / "summary.json").write_text(canonical_json(summary), encoding="utf-8")
        mlflow.log_artifacts(str(output_dir), artifact_path="results")
        mlflow.log_artifact(str(data_path), artifact_path="data")
        for relative_path in source_files:
            path = ROOT / relative_path
            parent = path.relative_to(ROOT).parent.as_posix()
            destination = "source" if parent == "." else f"source/{parent}"
            mlflow.log_artifact(str(path), artifact_path=destination)
    print(canonical_json(summary))
    print(f"Resultados guardados en: {output_dir}")
    return summary


if __name__ == "__main__":
    main()
