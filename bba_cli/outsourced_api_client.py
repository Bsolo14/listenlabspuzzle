from __future__ import annotations

import json
import io
import sys
import os
from typing import Dict, Optional

from .models import GameInit, NextPerson, RunningState, CompletedState

# Add the external simulator to the path
external_simulator_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'external', 'berghain-challenge')
sys.path.insert(0, external_simulator_path)

from simulation_engine import SimulationEngine


class OutsourcedApiClient:
    def __init__(self, scenario: int):
        self.scenario = scenario
        self.game_id = None
        self._setup_simulation()

    def _setup_simulation(self):
        """Set up the simulation engine with the appropriate scenario configuration"""
        # Scenario configurations (copied from external simulator)
        if self.scenario == 1:
            self.config = {
                "constraints": {
                    "young": 600,
                    "well_dressed": 600
                },
                "venue_capacity": 1000
            }
        elif self.scenario == 2:
            self.config = {
                "constraints": {
                    "techno_lover": 650,
                    "well_connected": 450,
                    "creative": 300,
                    "berlin_local": 750
                },
                "venue_capacity": 1000
            }
        elif self.scenario == 3:
            self.config = {
                "constraints": {
                    "underground_veteran": 500,
                    "international": 650,
                    "fashion_forward": 550,
                    "queer_friendly": 250,
                    "vinyl_collector": 200,
                    "german_speaker": 800
                },
                "venue_capacity": 1000
            }
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}")

        self.simulation = SimulationEngine(self.config, self.scenario)

    def new_game(self, scenario: int, player_id: str) -> GameInit:
        """Start a new game and return the game initialization data"""
        game_data = self.simulation.start_game()

        # Convert the response to our expected format
        constraints = [{"attribute": attr, "minCount": count}
                      for attr, count in self.config["constraints"].items()]

        # Get attribute statistics from the simulation engine
        attr_stats = self.simulation.person_generator.attribute_frequencies
        correlations = self.simulation.person_generator.correlations

        attribute_statistics = {
            "relativeFrequencies": attr_stats,
            "correlations": correlations
        }

        from .models import Constraint
        game_init = GameInit(
            gameId=game_data["gameId"],
            constraints=[Constraint(**c) for c in constraints],
            relativeFrequencies=attribute_statistics["relativeFrequencies"],
            correlations=attribute_statistics["correlations"]
        )

        self.game_id = game_data["gameId"]
        return game_init

    def decide_and_next(self, game_id: str, person_index: int, accept: Optional[bool]) -> RunningState | CompletedState:
        """Make a decision and get the next person"""
        if game_id != self.game_id:
            return CompletedState(status="failed", rejectedCount=0, nextPerson=None, reason="invalid game id")

        # Make the decision
        game_data = self.simulation.decide_and_next(accept)

        status = game_data.get("status")
        if status == "running":
            np = game_data["nextPerson"]
            next_person = NextPerson(personIndex=np["personIndex"], attributes=np["attributes"])
            return RunningState(
                status="running",
                admittedCount=game_data.get("admittedCount", self.simulation.admitted_count),
                rejectedCount=game_data["rejectedCount"],
                nextPerson=next_person,
            )
        else:
            # Game completed or failed
            result = CompletedState(
                status=status,
                rejectedCount=game_data.get("rejectedCount", 0),
                nextPerson=None,
                reason=game_data.get("reason")
            )
            # Use the external simulator's attribute counts format
            if "attributeCounts" in game_data:
                result.admittedAttributes = game_data["attributeCounts"]
            return result
