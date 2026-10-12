"""Comprobar que Projects conserva la corrida y la configuración del entrenamiento."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_project", ROOT / "scripts" / "train_project.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class MlflowProjectTests(unittest.TestCase):
    def test_direct_invocation_preserves_arguments(self):
        arguments = ["--max-depth", "1", "--seed", "17"]
        self.assertEqual(adapter.resolve_training_arguments(arguments, {}),
                         (arguments, None, None))

    def test_inherited_run_uses_its_store_and_experiment(self):
        environment = {"MLFLOW_RUN_ID": "project-run", "MLFLOW_TRACKING_URI": "sqlite:///parent.db"}
        with patch("mlflow.tracking.MlflowClient") as client_type:
            client = client_type.return_value
            client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(experiment_id="7"))
            client.get_experiment.return_value = SimpleNamespace(name="Parent experiment")
            arguments, run_id, tracking_uri = adapter.resolve_training_arguments(
                ["--tracking-uri", "sqlite:///wrong.db", "--experiment-name", "Wrong",
                 "--max-depth", "1", "--seed", "17"], environment,
            )
        parsed = adapter.train.parse_args(arguments)
        self.assertEqual(parsed.tracking_uri, "sqlite:///parent.db")
        self.assertEqual(parsed.experiment_name, "Parent experiment")
        self.assertEqual((parsed.max_depth, parsed.seed), (1, 17))
        self.assertEqual((run_id, tracking_uri), ("project-run", "sqlite:///parent.db"))
        client_type.assert_called_once_with(tracking_uri="sqlite:///parent.db")
        client.get_run.assert_called_once_with("project-run")
        client.get_experiment.assert_called_once_with("7")
        self.assertEqual(environment["MLFLOW_RUN_ID"], "project-run")

    def test_inherited_run_without_store_fails_before_training(self):
        with self.assertRaisesRegex(RuntimeError, "MLFLOW_TRACKING_URI"):
            adapter.resolve_training_arguments([], {"MLFLOW_RUN_ID": "project-run"})

    def test_descriptor_matches_cli_defaults(self):
        from mlflow.projects._project_spec import load_project
        from mlflow.utils.environment import _PythonEnv

        project = load_project(str(ROOT))
        entry = project.get_entry_point("main")
        defaults = adapter.train.parse_args([])
        self.assertEqual(set(project._entry_points), {"main"})
        for name in ["n_estimators", "max_depth", "min_samples_leaf", "seed"]:
            self.assertEqual(entry.parameters[name].default, getattr(defaults, name))
        self.assertIn("scripts/train_project.py", entry.compute_command({}, None))
        environment = _PythonEnv.from_yaml(ROOT / "python_env.yaml")
        self.assertEqual(environment.python, "3.13.7")
        self.assertEqual(environment.dependencies, ["-r requirements.txt"])


if __name__ == "__main__":
    unittest.main()
