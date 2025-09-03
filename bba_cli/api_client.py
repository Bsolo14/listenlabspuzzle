from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional

import requests

from .models import CompletedState, Constraint, GameInit, NextPerson, RunningState


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def new_game(self, scenario: int, player_id: str) -> GameInit:
        url = f"{self.base_url}/new-game"
        params = {"scenario": scenario, "playerId": player_id}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        constraints = [Constraint(**c) for c in data["constraints"]]
        rel = data["attributeStatistics"]["relativeFrequencies"]
        cors = data["attributeStatistics"]["correlations"]
        return GameInit(
            gameId=data["gameId"],
            constraints=constraints,
            relativeFrequencies=rel,
            correlations=cors,
        )

    def decide_and_next(
        self, game_id: str, person_index: int, accept: Optional[bool]
    ) -> RunningState | CompletedState:
        url = f"{self.base_url}/decide-and-next"
        params: Dict[str, Any] = {"gameId": game_id, "personIndex": person_index}
        if accept is not None:
            params["accept"] = str(accept).lower()
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "running":
            np = data["nextPerson"]
            next_person = NextPerson(personIndex=np["personIndex"], attributes=np["attributes"])
            return RunningState(
                status="running",
                admittedCount=data["admittedCount"],
                rejectedCount=data["rejectedCount"],
                nextPerson=next_person,
            )
        else:
            return CompletedState(status=status, rejectedCount=data["rejectedCount"], nextPerson=None)


