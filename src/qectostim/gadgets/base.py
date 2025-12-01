# src/qec_to_stim/gadgets/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import stim

from src.qectostim.codes.abstract_code import Code
from src.qectostim.noise.models import NoiseModel


class Gadget(ABC):
    """
    Abstract base for gadgets: logical operations implemented by specific protocols
    (teleportation, CSS surgery, transversal wrappers, etc.).
    """

    @abstractmethod
    def to_stim(self, codes: List[Code], noise_model: NoiseModel) -> stim.Circuit:
        """
        Build full Stim circuit implementing this gadget between the given code blocks.
        """
        ...