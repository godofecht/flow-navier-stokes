"""Bounded semantic execution trace for the actual Navier-Stokes solver.

Each event describes a mathematical operation actually performed by solver.py.
Intent labels describe the numerical method, not inferred author motivation.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SemanticEvent:
    step: int
    stage: str
    operation: str
    meaning: str
    diagnostic: dict

    def to_dict(self):
        return asdict(self)


def event(step, stage, operation, meaning, **diagnostic):
    return SemanticEvent(step, stage, operation, meaning,
                         {key: float(value) for key, value in diagnostic.items()})
