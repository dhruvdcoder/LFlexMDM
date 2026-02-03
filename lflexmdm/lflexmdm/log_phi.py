import lightning as L
import torch
import wandb
import numpy as np


class LogPhiToTensorBoard(L.pytorch.Callback):
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._log_params(
            outputs=outputs,
            pl_module=pl_module,
            trainer=trainer,
            group_name="phi_params",
        )
        self._log_params(
            outputs=outputs,
            pl_module=pl_module,
            trainer=trainer,
            group_name="theta_params",
        )

    def _log_params(
        self, outputs, pl_module, trainer, group_name="phi_params"
    ):
        if group_name in outputs and outputs[group_name] is not None:
            for name, value in outputs[group_name].items():
                if (
                    value is not None
                    and isinstance(value, torch.Tensor)
                    and value.ndim == 2
                ):  # (B, L)
                    for i in range(value.shape[0]):
                        # pick only non-nan values
                        val = value[i][~torch.isnan(value[i])]
                        if hasattr(
                            pl_module.logger.experiment, "add_histogram"
                        ):
                            # tensorboard logger
                            pl_module.logger.experiment.add_histogram(
                                f"{group_name}/{name}",
                                val,
                                trainer.global_step,
                            )
                        else:
                            # wandb logger
                            # skip if empty
                            n_bins = min(10, val.numel() - 3)
                            if n_bins <= 0:
                                continue

                            # Check if data has sufficient range for histogram
                            data_range = val.max() - val.min()
                            if data_range == 0 or not torch.isfinite(
                                data_range
                            ):
                                # All values are the same or infinite, skip histogram
                                continue

                            try:
                                hist = np.histogram(
                                    val.cpu().numpy(), bins=n_bins
                                )
                                pl_module.logger.experiment.log(
                                    {
                                        f"{group_name}/{name}": wandb.Histogram(
                                            np_histogram=hist,
                                            num_bins=n_bins,
                                        )
                                    },
                                    step=trainer.global_step,
                                )
                            except ValueError:
                                # Skip if histogram creation fails (e.g., insufficient data range)
                                pass
                        break  # log only one example
