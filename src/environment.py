"""
Battle Environment for Pokemon Showdown RL Training

This module provides a Gymnasium-compatible environment wrapper around poke-env
for training reinforcement learning agents on Pokemon battles.
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.player import Gen9EnvSinglePlayer
from poke_env.environment.abstract_battle import AbstractBattle


class Pokemon(np.ndarray):
    pass


# Type effectiveness chart (simplified - attacking type -> defending type -> multiplier)
TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}

# All Pokemon types
ALL_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

# Status conditions
STATUS_CONDITIONS = ["brn", "frz", "par", "psn", "slp", "tox"]


def get_type_effectiveness(attacking_type: str, defending_types: list) -> float:
    """Calculate type effectiveness multiplier."""
    if attacking_type is None:
        return 1.0

    attacking_type = attacking_type.lower()
    multiplier = 1.0

    for def_type in defending_types:
        if def_type is None:
            continue
        def_type = def_type.lower()
        if attacking_type in TYPE_CHART:
            multiplier *= TYPE_CHART[attacking_type].get(def_type, 1.0)

    return multiplier


class ShowdownEnv(Gen9EnvSinglePlayer):
    """
    Gymnasium-compatible environment for Pokemon Showdown battles.

    State space (~76 dimensions):
    - Active Pokemon: HP%, 18 type bits (one-hot), 6 status bits, 6 stat boosts = 31
    - Opponent active: HP%, 18 type bits, 6 status bits, 6 stat boosts = 31
    - Available moves: 4 moves * (base_power + type_effectiveness) = 8
    - Team HP: 6 Pokemon per side = 6

    Action space (22 discrete actions):
    - Actions 0-3: Use move 1-4
    - Actions 4-9: Switch to Pokemon 1-6 (in team order)
    - Actions 10-13: Use move 1-4 with mega evolution
    - Actions 14-17: Use move 1-4 with z-move
    - Actions 18-21: Use move 1-4 with dynamax/terastallize
    """

    # Observation space size
    OBS_SIZE = 76

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track previous state for reward shaping
        self._prev_own_hp = 1.0
        self._prev_opp_hp = 1.0
        self._prev_own_fainted = 0
        self._prev_opp_fainted = 0

    def calc_reward(self, last_battle, current_battle) -> float:
        """
        Calculate reward for the transition.

        Reward components:
        - Win: +1.0, Loss: -1.0
        - Damage dealt to opponent (scaled 0-0.1)
        - Damage received (penalty, scaled 0-0.1)
        - Knockout bonus: +0.1 per opponent fainted
        - Knockout penalty: -0.1 per own Pokemon fainted
        """
        return self.reward_computing_helper(
            current_battle,
            fainted_value=0.15,
            hp_value=0.15,
            victory_value=1.0,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Convert battle state to a feature vector for the RL agent.

        Returns a numpy array of shape (OBS_SIZE,) with values in [0, 1] or [-1, 1].
        """
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)
        idx = 0

        # === Active Pokemon (own) ===
        if battle.active_pokemon is not None:
            pokemon = battle.active_pokemon

            # HP percentage (0-1)
            obs[idx] = pokemon.current_hp_fraction
            idx += 1

            # Types (one-hot, 18 bits)
            for t in ALL_TYPES:
                if pokemon.type_1 is not None and pokemon.type_1.name.lower() == t:
                    obs[idx] = 1.0
                if pokemon.type_2 is not None and pokemon.type_2.name.lower() == t:
                    obs[idx] = 1.0
                idx += 1

            # Status condition (one-hot, 6 bits)
            if pokemon.status is not None:
                status_name = pokemon.status.name.lower()
                for i, s in enumerate(STATUS_CONDITIONS):
                    if status_name == s:
                        obs[idx + i] = 1.0
            idx += 6

            # Stat boosts (-6 to +6, normalized to -1 to 1)
            for stat in ["atk", "def", "spa", "spd", "spe", "accuracy"]:
                boost = pokemon.boosts.get(stat, 0)
                obs[idx] = boost / 6.0
                idx += 1
        else:
            idx += 31  # Skip active Pokemon features

        # === Opponent Active Pokemon ===
        if battle.opponent_active_pokemon is not None:
            pokemon = battle.opponent_active_pokemon

            # HP percentage (estimated for opponent)
            obs[idx] = pokemon.current_hp_fraction
            idx += 1

            # Types (one-hot, 18 bits)
            for t in ALL_TYPES:
                if pokemon.type_1 is not None and pokemon.type_1.name.lower() == t:
                    obs[idx] = 1.0
                if pokemon.type_2 is not None and pokemon.type_2.name.lower() == t:
                    obs[idx] = 1.0
                idx += 1

            # Status condition (one-hot, 6 bits)
            if pokemon.status is not None:
                status_name = pokemon.status.name.lower()
                for i, s in enumerate(STATUS_CONDITIONS):
                    if status_name == s:
                        obs[idx + i] = 1.0
            idx += 6

            # Stat boosts
            for stat in ["atk", "def", "spa", "spd", "spe", "accuracy"]:
                boost = pokemon.boosts.get(stat, 0)
                obs[idx] = boost / 6.0
                idx += 1
        else:
            idx += 31  # Skip opponent features

        # === Available Moves (4 moves) ===
        moves = battle.available_moves
        for i in range(4):
            if i < len(moves):
                move = moves[i]

                # Base power (normalized, 0-250 -> 0-1)
                obs[idx] = min(move.base_power / 250.0, 1.0)
                idx += 1

                # Type effectiveness against opponent
                if battle.opponent_active_pokemon is not None:
                    opp_types = []
                    if battle.opponent_active_pokemon.type_1:
                        opp_types.append(battle.opponent_active_pokemon.type_1.name)
                    if battle.opponent_active_pokemon.type_2:
                        opp_types.append(battle.opponent_active_pokemon.type_2.name)

                    effectiveness = get_type_effectiveness(
                        move.type.name if move.type else None,
                        opp_types
                    )
                    # Normalize: 0=immune, 0.25=0.25x, 0.5=0.5x, 1=1x, 2=2x, 4=4x
                    obs[idx] = min(effectiveness / 4.0, 1.0)
                else:
                    obs[idx] = 0.25  # Unknown effectiveness
                idx += 1
            else:
                idx += 2  # No move available

        # === Team HP (own team, 6 Pokemon) ===
        team_pokemon = list(battle.team.values())
        for i in range(6):
            if i < len(team_pokemon):
                obs[idx] = team_pokemon[i].current_hp_fraction
            idx += 1

        return obs

    def describe_embedding(self) -> dict:
        """Return a description of the observation space for debugging."""
        return {
            "total_size": self.OBS_SIZE,
            "active_pokemon": {
                "hp": 1,
                "types": 18,
                "status": 6,
                "boosts": 6,
                "total": 31
            },
            "opponent_active": {
                "hp": 1,
                "types": 18,
                "status": 6,
                "boosts": 6,
                "total": 31
            },
            "moves": {
                "per_move": 2,  # base_power, type_effectiveness
                "num_moves": 4,
                "total": 8
            },
            "team_hp": 6
        }


class SimpleOpponent(Gen9EnvSinglePlayer):
    """A simple opponent that picks moves based on base power."""

    def embed_battle(self, battle):
        return np.zeros(1)

    def calc_reward(self, last_battle, current_battle):
        return 0


def create_environment(
    battle_format: str = "gen9randombattle",
    server_configuration=None,
    opponent_type: str = "random"
):
    """
    Factory function to create a training environment.

    Args:
        battle_format: Pokemon Showdown format (e.g., "gen9randombattle")
        server_configuration: poke-env server configuration (None for local)
        opponent_type: Type of opponent ("random", "max_damage", "self")

    Returns:
        Tuple of (environment, opponent_player)
    """
    from poke_env import LocalhostServerConfiguration
    from poke_env.player import RandomPlayer, MaxBasePowerPlayer

    if server_configuration is None:
        server_configuration = LocalhostServerConfiguration

    # Create the RL environment
    env_player = ShowdownEnv(
        battle_format=battle_format,
        server_configuration=server_configuration,
        start_challenging=False,
    )

    # Create opponent based on type
    if opponent_type == "random":
        opponent = RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_configuration,
        )
    elif opponent_type == "max_damage":
        opponent = MaxBasePowerPlayer(
            battle_format=battle_format,
            server_configuration=server_configuration,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    return env_player, opponent
