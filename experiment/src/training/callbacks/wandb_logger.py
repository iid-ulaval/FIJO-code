from typing import Dict, List
import os

import wandb
from poutyne.framework.callbacks import Logger


class WandbLogger(Logger):
    def __init__(self,
                 project_name: str,
                 groupe_name: str = None,
                 config: dict = None,
                 initial_artifacts_paths: List = None,
                 run_id: str = None,
                 batch_granularity: bool = False,
                 checkpoints_path: str = None):
        super().__init__(batch_granularity=batch_granularity)

        self.checkpoints_path = checkpoints_path

        wandb.init(project=project_name,
                   group=groupe_name,
                   config=config,
                   resume="allow",
                   id=run_id,
                   reinit=False)

        wandb.config.update({"run_id": wandb.run.id})

        if initial_artifacts_paths is not None:
            self._log_artifacts(initial_artifacts_paths,
                                name="Initial-artifacts",
                                artifact_type="Miscellaneous")

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        logs.pop("epoch")
        logs.pop("time")

        train_metrics = {
            key: value
            for (key, value) in logs.items() if "val_" not in key
        }
        val_metrics = {
            key: value
            for (key, value) in logs.items() if "val_" in key
        }

        train_metrics = {"training": train_metrics}
        val_metrics = {"validation": val_metrics}

        learning_rate = self._get_current_learning_rates()
        self._log_metrics(train_metrics, step=epoch_number)
        self._log_metrics(val_metrics, step=epoch_number)
        self._log_params(learning_rate, step=epoch_number)

    def _on_train_end_write(self, logs: Dict):
        if self.checkpoints_path is not None:
            self._log_artifacts([self.checkpoints_path],
                                "Checkpoints",
                                artifact_type="Model-weights")

    def on_test_end(self, logs: Dict):
        self._on_test_end_write(logs)
        wandb.finish()

    def _on_test_end_write(self, logs: Dict):
        logs = {"testing": logs}
        self._log_metrics(logs, step=wandb.run.step + 1)

    def _log_metrics(self, metrics: Dict, step: int):
        wandb.log(metrics, step=step)

    def _log_params(self, params: Dict, step: int):
        wandb.log({"params": params}, step=step)

    def _log_artifacts(self, paths: List[str], name: str, artifact_type: str):
        artifact = wandb.Artifact(name=name, type=artifact_type)
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"The path {path} is not a file nor a directory")

            if os.path.isdir(path):
                artifact.add_dir(path)
            elif os.path.isfile(path):
                artifact.add_file(path)

        wandb.log_artifact(artifact)
