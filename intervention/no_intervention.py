from __future__ import annotations

from domain.content import DebunkPost

from .base import InterventionStrategy


class NoIntervention(InterventionStrategy):
    def select_targets(self, sim_state) -> list[str]:
        return []

    def generate_interventions(self, targets: list[str], sim_state) -> list[DebunkPost]:
        return []
