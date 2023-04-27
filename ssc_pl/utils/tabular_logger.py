import logging
import os
from typing import Dict, Optional

from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.csv_logs import CSVLogger, ExperimentWriter

log = logging.getLogger(__name__)


class ConsoleExperimentWriter(ExperimentWriter):
    r"""
    Experiment writer for CSVLogger.

    Currently supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_METRICS_FILE = 'log.txt'

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""
        super().log_metrics(metrics_dict, step)
        metrics = self.metrics[-1]

        def metrics2str(metrics: dict) -> str:
            return ', '.join([
                f'{x}: {{{metrics2str(y)}}}' if isinstance(y, dict) else
                (f'{x}: {y:.4f}' if y >= 1e-4 else f'{x}: {y:.5f}')
                if isinstance(y, float) else f'{x}: {y}' for x, y in metrics.items()
            ])

        log_str = metrics2str(metrics)
        with open(self.metrics_file_path, 'a+') as f:
            f.write(log_str + '\n')
        log.info(log_str)

    def save(self) -> None:
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)


class TabularLogger(CSVLogger):

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~lightning.pytorch.core.module.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ConsoleExperimentWriter(log_dir=self.log_dir)
        return self._experiment
