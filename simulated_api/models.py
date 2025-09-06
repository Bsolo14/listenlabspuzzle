from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal

from bba_cli.models import AttributeId, Constraint, GameInit, NextPerson, RunningState, CompletedState


@dataclass
class SimulatedGame:
    game_id: str
    scenario: int
    constraints: List[Constraint]
    relative_frequencies: Mapping[AttributeId, float]
    correlations: Mapping[AttributeId, Mapping[AttributeId, float]]
    attribute_order: List[AttributeId]
    admitted_count: int = 0
    rejected_count: int = 0
    person_index: int = 0
    admitted_attributes: Dict[AttributeId, int] = None

    def __post_init__(self):
        if self.admitted_attributes is None:
            self.admitted_attributes = {attr: 0 for attr in self.relative_frequencies.keys()}

    def is_complete(self) -> bool:
        """Check if the game is complete (venue full or too many rejections)"""
        return self.admitted_count >= 1000 or self.rejected_count >= 20000

    def check_constraints_satisfied(self) -> bool:
        """Check if all constraints are satisfied"""
        for constraint in self.constraints:
            if self.admitted_attributes[constraint.attribute] < constraint.minCount:
                return False
        return True

    def get_status(self) -> str:
        """Get the current game status"""
        if self.is_complete():
            if self.admitted_count >= 1000 and self.check_constraints_satisfied():
                return "completed"
            else:
                return "failed"
        return "running"


class PersonGenerator:
    """Generates people based on attribute correlations using Gaussian copula"""

    def __init__(self, relative_frequencies: Mapping[AttributeId, float],
                 correlations: Mapping[AttributeId, Mapping[AttributeId, float]],
                 attribute_order: List[AttributeId]):
        self.relative_frequencies = relative_frequencies
        self.correlations = correlations
        self.attribute_order = attribute_order
        self.cov_matrix = self._build_covariance_matrix()

    def _build_covariance_matrix(self) -> np.ndarray:
        """Build covariance matrix from correlations"""
        n = len(self.attribute_order)
        cov = np.eye(n)

        for i, attr1 in enumerate(self.attribute_order):
            for j, attr2 in enumerate(self.attribute_order):
                if i != j:
                    cov[i, j] = self.correlations[attr1][attr2]

        return cov

    def generate_person(self) -> Dict[AttributeId, bool]:
        """Generate a person with correlated binary attributes"""
        # Use Gaussian copula to generate correlated uniforms
        n_attrs = len(self.attribute_order)
        mean = np.zeros(n_attrs)

        # Handle the case where covariance matrix might be singular
        try:
            normals = multivariate_normal.rvs(mean=mean, cov=self.cov_matrix, size=1)
            # Convert to uniforms using normal CDF (scipy.stats.norm.cdf)
            from scipy.stats import norm
            uniforms = norm.cdf(normals)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to independent generation if covariance matrix is problematic
            uniforms = np.random.random(n_attrs)

        # Convert uniforms to binary attributes based on relative frequencies
        attributes = {}
        for i, attr in enumerate(self.attribute_order):
            threshold = self.relative_frequencies[attr]
            attributes[attr] = bool(uniforms[i] < threshold)

        return attributes


class ScenarioData:
    """Manages scenario data loaded from scenarios.json"""

    def __init__(self, scenarios_file: str = "/Users/barrowsolomon/listen_labs_puzzle/scenarios.json"):
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
        self.scenarios = data['scenarios']

    def get_scenario(self, scenario_id: int) -> Dict:
        """Get scenario data by ID"""
        return self.scenarios[str(scenario_id)]

    def create_game_init(self, scenario_id: int) -> GameInit:
        """Create GameInit object for a scenario"""
        scenario = self.get_scenario(scenario_id)
        constraints = [Constraint(**c) for c in scenario["constraints"]]
        rel_freq = scenario["attributeStatistics"]["relativeFrequencies"]
        correlations = scenario["attributeStatistics"]["correlations"]

        return GameInit(
            gameId=str(uuid.uuid4()),
            constraints=constraints,
            relativeFrequencies=rel_freq,
            correlations=correlations,
        )
