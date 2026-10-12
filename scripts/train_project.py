"""Adaptar MLflow Projects al entrenamiento existente sin crear otra corrida."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402


def resolve_training_arguments(argv=None, environ=None):
    """Usar el almacén y experimento de la corrida abierta por MLflow Projects."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if environ is None else environ
    run_id = environment.get("MLFLOW_RUN_ID")
    if not run_id:
        return arguments, None, None

    tracking_uri = environment.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_RUN_ID requiere el MLFLOW_TRACKING_URI de MLflow Projects.")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    if experiment is None:
        raise RuntimeError("No se encontró el experimento de la corrida de MLflow Projects.")

    # La última aparición prevalece en argparse: no se puede cambiar el almacén
    # o experimento de una corrida que Projects ya creó.
    arguments.extend([
        "--tracking-uri", tracking_uri,
        "--experiment-name", experiment.name,
    ])
    return arguments, run_id, tracking_uri


def main(argv=None):
    arguments, inherited_run_id, tracking_uri = resolve_training_arguments(argv)
    # start_run() dentro de train.main consume MLFLOW_RUN_ID y reanuda esa corrida.
    summary = train.main(arguments)
    if inherited_run_id and summary["run_id"] != inherited_run_id:
        raise AssertionError("El entrenamiento no reutilizó la corrida de MLflow Projects.")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri or train.parse_args(arguments).tracking_uri)
    for relative_path in ["MLproject", "python_env.yaml", "scripts/train_project.py"]:
        client.log_artifact(summary["run_id"], str(ROOT / relative_path), "project_source")
    client.set_tag(summary["run_id"], "execution_interface",
                   "mlflow_projects" if inherited_run_id else "project_adapter_cli")
    return summary


if __name__ == "__main__":
    main()
