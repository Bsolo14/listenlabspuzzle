from __future__ import annotations

import uuid
from typing import Dict, Optional

from flask import Flask, request, jsonify

from models import PersonGenerator, ScenarioData, SimulatedGame


class SimulatedAPIServer:
    def __init__(self, quiet: bool = False):
        self.app = Flask(__name__)
        self.quiet = quiet
        self.scenario_data = ScenarioData()
        self.active_games: Dict[str, SimulatedGame] = {}

        # Disable Flask request logging if quiet
        if quiet:
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/new-game', methods=['GET'])
        def new_game():
            scenario = int(request.args.get('scenario'))
            player_id = request.args.get('playerId')

            if scenario not in [1, 2, 3]:
                return jsonify({"error": "Invalid scenario"}), 400

            # Create game init data
            game_init = self.scenario_data.create_game_init(scenario)

            # Create simulated game state
            attribute_order = list(game_init.relativeFrequencies.keys())
            simulated_game = SimulatedGame(
                game_id=game_init.gameId,
                scenario=scenario,
                constraints=game_init.constraints,
                relative_frequencies=game_init.relativeFrequencies,
                correlations=game_init.correlations,
                attribute_order=attribute_order,
            )

            # Store the game
            self.active_games[game_init.gameId] = simulated_game

            # Return the same format as real API
            response = {
                "gameId": game_init.gameId,
                "constraints": [
                    {"attribute": c.attribute, "minCount": c.minCount}
                    for c in game_init.constraints
                ],
                "attributeStatistics": {
                    "relativeFrequencies": dict(game_init.relativeFrequencies),
                    "correlations": dict(game_init.correlations)
                }
            }

            return jsonify(response)

        @self.app.route('/decide-and-next', methods=['GET'])
        def decide_and_next():
            try:
                game_id = request.args.get('gameId')
                person_index = int(request.args.get('personIndex'))
                accept_str = request.args.get('accept')
                accept = accept_str.lower() == 'true' if accept_str else None

                if game_id not in self.active_games:
                    return jsonify({"error": "Game not found"}), 404

                game = self.active_games[game_id]

                # Validate person index
                if person_index != game.person_index:
                    return jsonify({"error": "Invalid person index"}), 400

                # For person_index=0, accept parameter is optional (no decision needed)
                if person_index > 0 and accept is None:
                    return jsonify({"error": "Accept parameter required for person_index > 0"}), 400

                # Make decision if accept is provided
                if accept is not None:
                    if accept:
                        game.admitted_count += 1
                        # Update admitted attributes
                        person_attrs = game.current_person_attributes
                        for attr, has_attr in person_attrs.items():
                            if has_attr:
                                game.admitted_attributes[attr] += 1
                    else:
                        game.rejected_count += 1

                # Generate next person
                if not hasattr(game, 'person_generator'):
                    game.person_generator = PersonGenerator(
                        game.relative_frequencies,
                        game.correlations,
                        game.attribute_order
                    )

                next_person_attrs = game.person_generator.generate_person()
                game.current_person_attributes = next_person_attrs
                game.person_index += 1

                # Check game status
                status = game.get_status()

                if status == "running":
                    response = {
                        "status": "running",
                        "admittedCount": game.admitted_count,
                        "rejectedCount": game.rejected_count,
                        "nextPerson": {
                            "personIndex": game.person_index,
                            "attributes": next_person_attrs
                        }
                    }
                else:
                    # Build terminal response according to API spec
                    response = {
                        "status": status,
                        "nextPerson": None
                    }

                    if status == "completed":
                        response["rejectedCount"] = game.rejected_count
                        # Include attribute counts for analysis
                        response["admittedAttributes"] = dict(game.admitted_attributes)
                    elif status == "failed":
                        # Provide a reason for failure per spec
                        if game.rejected_count >= 20000:
                            reason = "rejection cap reached"
                        elif game.admitted_count >= 1000 and not game.check_constraints_satisfied():
                            reason = "constraints not satisfied at capacity"
                        else:
                            reason = "game ended in failed state"
                        response["reason"] = reason
                        # Include attribute counts for analysis even on failure
                        response["admittedAttributes"] = dict(game.admitted_attributes)

                    # Clean up completed game
                    del self.active_games[game_id]

                return jsonify(response)
            except Exception as e:
                print(f"Error in decide-and-next: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500

    def run(self, host: str = 'localhost', port: int = 5000, debug: bool = True):
        """Run the simulated API server"""
        if not self.quiet:
            print(f"Starting simulated API server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug and not self.quiet)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Simulated API Server")
    parser.add_argument("--quiet", action="store_true", help="Reduce server output and logging")
    args = parser.parse_args()

    server = SimulatedAPIServer(quiet=args.quiet)
    server.run()
