#!/usr/bin/env python3
"""
Test script for the simulated API
"""

import requests
import time


def test_simulated_api():
    """Test the simulated API endpoints"""
    base_url = "http://localhost:5000"

    # Test new game
    print("Testing new-game endpoint...")
    response = requests.get(f"{base_url}/new-game?scenario=1&playerId=test")
    if response.status_code != 200:
        print(f"Failed to create game: {response.status_code}")
        return

    game_data = response.json()
    game_id = game_data['gameId']
    print(f"Created game: {game_id}")

    # Test first person (no decision needed)
    print("Getting first person...")
    response = requests.get(f"{base_url}/decide-and-next?gameId={game_id}&personIndex=0")
    if response.status_code != 200:
        print(f"Failed to get first person: {response.status_code}")
        return

    person_data = response.json()
    print(f"First person: {person_data['nextPerson']}")

    # Test a few decisions
    person_index = person_data['nextPerson']['personIndex']
    for i in range(3):
        # Accept the person
        response = requests.get(f"{base_url}/decide-and-next?gameId={game_id}&personIndex={person_index}&accept=true")
        if response.status_code != 200:
            print(f"Failed to make decision: {response.status_code}")
            return

        decision_data = response.json()
        print(f"Decision {i+1}: admitted={decision_data['admittedCount']}, rejected={decision_data['rejectedCount']}")

        if decision_data['status'] != 'running':
            print(f"Game ended: {decision_data['status']}")
            break

        person_index = decision_data['nextPerson']['personIndex']

    print("Test completed successfully!")


if __name__ == '__main__':
    test_simulated_api()
