from __future__ import annotations

from typing import Dict, Optional

from .models import GameInit, NextPerson, RunningState, CompletedState
from simulated_api.models import ScenarioData, SimulatedGame, PersonGenerator


class EmbeddedApiClient:
    def __init__(self):
        self.scenario_data = ScenarioData()
        self._active_games: Dict[str, SimulatedGame] = {}

    def new_game(self, scenario: int, player_id: str) -> GameInit:
        game_init = self.scenario_data.create_game_init(scenario)
        attribute_order = list(game_init.relativeFrequencies.keys())
        self._active_games[game_init.gameId] = SimulatedGame(
            game_id=game_init.gameId,
            scenario=scenario,
            constraints=game_init.constraints,
            relative_frequencies=game_init.relativeFrequencies,
            correlations=game_init.correlations,
            attribute_order=attribute_order,
        )
        return game_init

    def decide_and_next(self, game_id: str, person_index: int, accept: Optional[bool]) -> RunningState | CompletedState:
        game = self._active_games.get(game_id)
        if game is None:
            return CompletedState(status="failed", rejectedCount=0, nextPerson=None, reason="game not found")

        if person_index != game.person_index:
            return CompletedState(status="failed", rejectedCount=game.rejected_count, nextPerson=None, reason="invalid person index")

        if accept is not None:
            if accept:
                game.admitted_count += 1
                for attr, has_attr in game.current_person_attributes.items():
                    if has_attr:
                        game.admitted_attributes[attr] += 1
            else:
                game.rejected_count += 1

        if getattr(game, "person_generator", None) is None:
            game.person_generator = PersonGenerator(game.relative_frequencies, game.correlations, game.attribute_order)

        next_attrs = game.person_generator.generate_person()
        game.current_person_attributes = next_attrs
        game.person_index += 1

        status = game.get_status()

        if status == "running":
            return RunningState(
                status="running",
                admittedCount=game.admitted_count,
                rejectedCount=game.rejected_count,
                nextPerson=NextPerson(personIndex=game.person_index, attributes=next_attrs),
            )

        if status == "completed":
            result = CompletedState(status="completed", rejectedCount=game.rejected_count, nextPerson=None)
            result.admittedAttributes = dict(game.admitted_attributes)
        else:
            if game.rejected_count >= 20000:
                reason = "rejection cap reached"
            elif game.admitted_count >= 1000 and not game.check_constraints_satisfied():
                reason = "constraints not satisfied at capacity"
            else:
                reason = "game ended in failed state"
            result = CompletedState(status="failed", rejectedCount=game.rejected_count, nextPerson=None, reason=reason)
            result.admittedAttributes = dict(game.admitted_attributes)

        del self._active_games[game_id]
        return result
