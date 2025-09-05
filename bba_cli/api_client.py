from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Mapping, Optional

import requests

from .models import CompletedState, Constraint, GameInit, NextPerson, RunningState


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _make_request_with_retry(self, url: str, params: Dict[str, Any], max_retries: int = 3, retry_delay: float = 2.5) -> Dict[str, Any]:
        """Make HTTP request with retry logic for handling network timeouts and disconnections."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = retry_delay * (attempt + 1)  # Progressive delay
                    print(f"[warn] Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"[info] Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[error] All {max_retries + 1} attempts failed. Last error: {e}")
                    raise e

        # This should never be reached, but just in case
        raise last_exception

    def new_game(self, scenario: int, player_id: str) -> GameInit:
        url = f"{self.base_url}/new-game"
        params = {"scenario": scenario, "playerId": player_id}
        data = self._make_request_with_retry(url, params)
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
        data = self._make_request_with_retry(url, params)
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


