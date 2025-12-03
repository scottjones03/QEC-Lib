from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Decoder(ABC):
    """Minimal decoder interface used by Experiment.run_decode."""

    @abstractmethod
    def decode_batch(self, dets: np.ndarray) -> Any:
        ...
