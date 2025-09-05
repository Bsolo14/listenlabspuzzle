from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple, Union, Optional


AttributeId = str
TypeKey = Tuple[int, ...]  # encode binary attributes as 0/1 tuple order-aligned with attribute list


@dataclass
class Constraint:
    attribute: AttributeId
    minCount: int


@dataclass
class GameInit:
    gameId: str
    constraints: List[Constraint]
    relativeFrequencies: Mapping[AttributeId, float]
    correlations: Mapping[AttributeId, Mapping[AttributeId, float]]


@dataclass
class NextPerson:
    personIndex: int
    attributes: Mapping[AttributeId, bool]


@dataclass
class RunningState:
    status: str
    admittedCount: int
    rejectedCount: int
    nextPerson: NextPerson


@dataclass
class CompletedState:
    status: str
    rejectedCount: int
    nextPerson: None
    reason: Optional[str] = None
    admittedAttributes: Optional[Dict[AttributeId, int]] = None


ApiState = Union[RunningState, CompletedState]


