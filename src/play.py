"""
Real-Time Play Interface for Pokemon Showdown RL Agent

Connect a trained model to Pokemon Showdown for real battles.
Can play on the public server or accept challenges locally.

Usage:
    # Accept challenges on local server
    python -m src.play --model models/selfplay_final --local

    # Play on public server (requires account)
    python -m src.play --model models/selfplay_final --username MyBot --password secret

    # Ladder matches
    python -m src.play --model models/selfplay_final --username MyBot --password secret --ladder 10
"""

import argparse
import asyncio
from pathlib import Path

from stable_baselines3 import PPO

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.player import Player

from src.environment import ShowdownEnv


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class TrainedPlayer(Player):
    """
    A Pokemon Showdown player that uses a trained RL model for decisions.

    This class can connect to both local and public Showdown servers
    and play battles using the trained model.
    """

    def __init__(
        self,
        model_path: str,
        battle_format: str = "gen9randombattle",
        account_configuration: AccountConfiguration = None,
        server_configuration=None,
        **kwargs
    ):
        super().__init__(
            battle_format=battle_format,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            **kwargs
        )

        # Load the trained model
        print(f"Loading model from {model_path}...")
        # We need a dummy env to load the model
        self._dummy_env = ShowdownEnv(
            battle_format=battle_format,
            server_configuration=server_configuration or LocalhostServerConfiguration,
            start_challenging=False,
        )
        self._model = PPO.load(model_path, env=self._dummy_env)
        print("Model loaded!")

    def embed_battle(self, battle):
        """Use the same embedding as training environment."""
        return self._dummy_env.embed_battle(battle)

    def choose_move(self, battle):
        """
        Use the trained model to select the best action.

        This method is called by poke-env whenever it's our turn to act.
        """
        # Get observation from battle state
        obs = self.embed_battle(battle)

        # Get action from model (deterministic for playing)
        action, _ = self._model.predict(obs, deterministic=True)

        # Convert action index to actual move
        return self._action_to_move(action, battle)

    def _action_to_move(self, action: int, battle):
        """Convert model action to poke-env move order."""
        # Actions 0-3: Use one of the 4 available moves
        if action < 4:
            if action < len(battle.available_moves):
                move = battle.available_moves[action]
                return self.create_order(move)

        # Actions 4-9: Switch to one of the team members
        elif action < 10:
            switch_idx = action - 4
            available_switches = battle.available_switches
            if switch_idx < len(available_switches):
                return self.create_order(available_switches[switch_idx])

        # Actions 10-13: Move with mega evolution
        elif action < 14:
            move_idx = action - 10
            if battle.can_mega_evolve and move_idx < len(battle.available_moves):
                return self.create_order(
                    battle.available_moves[move_idx],
                    mega=True
                )

        # Actions 14-17: Move with Z-move
        elif action < 18:
            move_idx = action - 14
            if battle.can_z_move and move_idx < len(battle.available_moves):
                return self.create_order(
                    battle.available_moves[move_idx],
                    z_move=True
                )

        # Actions 18-21: Move with dynamax/terastallize
        elif action < 22:
            move_idx = action - 18
            if move_idx < len(battle.available_moves):
                if battle.can_dynamax:
                    return self.create_order(
                        battle.available_moves[move_idx],
                        dynamax=True
                    )
                elif battle.can_tera:
                    return self.create_order(
                        battle.available_moves[move_idx],
                        terastallize=True
                    )

        # Fallback: Choose a random valid move
        return self.choose_random_move(battle)


async def accept_local_challenges(
    model_path: str,
    battle_format: str,
    n_challenges: int = None,
):
    """
    Accept challenges on a local Pokemon Showdown server.

    Args:
        model_path: Path to the trained model
        battle_format: Battle format to play
        n_challenges: Number of challenges to accept (None = infinite)
    """
    player = TrainedPlayer(
        model_path=model_path,
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    print(f"\nBot is ready!")
    print(f"Username: {player.username}")
    print(f"Format: {battle_format}")
    print(f"\nWaiting for challenges...")
    print("To challenge this bot, go to the local showdown and challenge:")
    print(f"  /challenge {player.username}")
    print("\nPress Ctrl+C to stop.\n")

    if n_challenges is None:
        # Accept challenges indefinitely
        while True:
            try:
                await player.accept_challenges(None, 1)
                print(f"Battle finished! Wins: {player.n_won_battles}/{player.n_finished_battles}")
            except asyncio.CancelledError:
                break
    else:
        await player.accept_challenges(None, n_challenges)
        print(f"\nFinished {n_challenges} battles.")
        print(f"Win rate: {player.n_won_battles}/{player.n_finished_battles}")


async def play_on_ladder(
    model_path: str,
    username: str,
    password: str,
    battle_format: str,
    n_games: int,
):
    """
    Play ladder matches on the public Pokemon Showdown server.

    Args:
        model_path: Path to the trained model
        username: Pokemon Showdown account username
        password: Pokemon Showdown account password
        battle_format: Battle format to play
        n_games: Number of ladder games to play
    """
    account_config = AccountConfiguration(username, password)

    player = TrainedPlayer(
        model_path=model_path,
        battle_format=battle_format,
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
    )

    print(f"\nConnecting to Pokemon Showdown as {username}...")
    print(f"Format: {battle_format}")
    print(f"Games: {n_games}")
    print("\nSearching for ladder matches...")

    await player.ladder(n_games)

    print(f"\nFinished {n_games} ladder games!")
    print(f"Win rate: {player.n_won_battles}/{player.n_finished_battles}")


async def challenge_player(
    model_path: str,
    username: str,
    password: str,
    opponent: str,
    battle_format: str,
    n_challenges: int,
):
    """
    Challenge a specific player on Pokemon Showdown.

    Args:
        model_path: Path to the trained model
        username: Your Pokemon Showdown account username
        password: Your Pokemon Showdown account password
        opponent: Username of the player to challenge
        battle_format: Battle format to play
        n_challenges: Number of challenges to send
    """
    account_config = AccountConfiguration(username, password)

    player = TrainedPlayer(
        model_path=model_path,
        battle_format=battle_format,
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
    )

    print(f"\nConnecting to Pokemon Showdown as {username}...")
    print(f"Challenging: {opponent}")
    print(f"Format: {battle_format}")

    await player.send_challenges(opponent, n_challenges)

    print(f"\nFinished challenging {opponent}!")
    print(f"Win rate: {player.n_won_battles}/{player.n_finished_battles}")


def main():
    parser = argparse.ArgumentParser(
        description="Play Pokemon Showdown with a trained RL agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (without .zip)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen9randombattle",
        help="Battle format (default: gen9randombattle)"
    )

    # Server options
    server_group = parser.add_mutually_exclusive_group()
    server_group.add_argument(
        "--local",
        action="store_true",
        help="Play on local server (default)"
    )
    server_group.add_argument(
        "--public",
        action="store_true",
        help="Play on public Pokemon Showdown server"
    )

    # Account configuration (for public server)
    parser.add_argument(
        "--username",
        type=str,
        help="Pokemon Showdown username (required for public server)"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Pokemon Showdown password (required for public server)"
    )

    # Play mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ladder",
        type=int,
        metavar="N",
        help="Play N ladder games"
    )
    mode_group.add_argument(
        "--challenge",
        type=str,
        metavar="OPPONENT",
        help="Challenge a specific player"
    )
    mode_group.add_argument(
        "--accept",
        type=int,
        nargs="?",
        const=0,
        metavar="N",
        help="Accept N challenges (0 or omit for infinite)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.public and not (args.username and args.password):
        print("Error: --username and --password required for public server")
        return

    # Determine server mode
    use_public = args.public or (args.ladder is not None) or (args.challenge is not None)

    if use_public and not (args.username and args.password):
        print("Error: --username and --password required for ladder/challenge mode")
        return

    # Run appropriate mode
    if args.ladder:
        asyncio.run(play_on_ladder(
            model_path=args.model,
            username=args.username,
            password=args.password,
            battle_format=args.format,
            n_games=args.ladder,
        ))
    elif args.challenge:
        asyncio.run(challenge_player(
            model_path=args.model,
            username=args.username,
            password=args.password,
            opponent=args.challenge,
            battle_format=args.format,
            n_challenges=1,
        ))
    else:
        # Default: accept challenges on local server
        n_challenges = None if args.accept == 0 else args.accept
        asyncio.run(accept_local_challenges(
            model_path=args.model,
            battle_format=args.format,
            n_challenges=n_challenges,
        ))


if __name__ == "__main__":
    main()
