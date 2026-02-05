from typing import List
from omegaconf import DictConfig
from lightning import Callback
from xlm.utils.rank_zero import RankedLogger
import hydra

logger = RankedLogger(__name__, rank_zero_only=True)


class CallbacksCreator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, cfg: DictConfig) -> List[Callback]:
        callbacks: List[Callback] = []
        for _, cb_conf in cfg.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
        return callbacks
