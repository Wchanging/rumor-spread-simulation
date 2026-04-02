from __future__ import annotations

from abc import ABC, abstractmethod

from domain.content import DebunkPost


class InterventionStrategy(ABC):
    @abstractmethod
    def select_targets(self, sim_state) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def generate_interventions(self, targets: list[str], sim_state) -> list[DebunkPost]:
        raise NotImplementedError
