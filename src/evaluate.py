"""
Evaluation Script for Pokemon Showdown RL Agent

Evaluates trained models against various baseline opponents
and reports win rates.

Usage:
    python -m src.evaluate --model models/selfplay_final_20231215_120000
    python -m src.evaluate --model models/stage1_final_20231215_120000 --battles 100
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from stable_baselines3 import PPO

from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from src.environment import ShowdownEnv


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class RLPlayer(ShowdownEnv):
    """Wrapper that uses a trained model for action selection."""

    def __init__(self, model: PPO, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = model

    def choose_move(self, battle):
        """Use the trained model to select an action."""
        obs = self.embed_battle(battle)
        action, _ = self._model.predict(obs, deterministic=True)

        # Convert action to move
        return self.action_to_move(action, battle)

    def action_to_move(self, action: int, battle):
        """Convert model action to poke-env move."""
        # Actions 0-3: Use move
        if action < 4:
            if action < len(battle.available_moves):
                return self.create_order(battle.available_moves[action])

        # Actions 4-9: Switch to Pokemon
        elif action < 10:
            switch_idx = action - 4
            available_switches = [
                p for p in battle.available_switches
                if not p.fainted
            ]
            if switch_idx < len(available_switches):
                return self.create_order(available_switches[switch_idx])

        # Fallback: random move
        return self.choose_random_move(battle)


async def evaluate_against_opponent(
    model: PPO,
    opponent_type: str,
    n_battles: int,
    battle_format: str,
) -> Dict[str, float]:
    """
    Evaluate a trained model against a specific opponent.

    Returns:
        Dictionary with win_rate, avg_turns, and other metrics.
    """
    server_config = LocalhostServerConfiguration
    timestamp = datetime.now().strftime("%H%M%S%f")

    # Create RL player with trained model
    player = RLPlayer(
        model=model,
        battle_format=battle_format,
        server_configuration=server_config,
    )

    # Create opponent
    if opponent_type == "random":
        opponent = RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
        )
    elif opponent_type == "max_damage":
        opponent = MaxBasePowerPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
        )
    elif opponent_type == "heuristics":
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    print(f"  Evaluating vs {opponent_type}... ", end="", flush=True)

    # Run battles
    await player.battle_against(opponent, n_battles=n_battles)

    # Calculate metrics
    wins = player.n_won_battles
    total = player.n_finished_battles
    win_rate = wins / total if total > 0 else 0

    # Calculate average turns from battle history
    total_turns = sum(
        len(b.observations) for b in player.battles.values()
    )
    avg_turns = total_turns / total if total > 0 else 0

    print(f"{win_rate:.1%} ({wins}/{total})")

    return {
        "opponent": opponent_type,
        "wins": wins,
        "total": total,
        "win_rate": win_rate,
        "avg_turns": avg_turns,
    }


async def full_evaluation(
    model_path: str,
    n_battles: int = 100,
    battle_format: str = "gen9randombattle",
) -> List[Dict]:
    """
    Run full evaluation against all baseline opponents.

    Args:
        model_path: Path to the trained model
        n_battles: Number of battles per opponent
        battle_format: Pokemon Showdown format

    Returns:
        List of evaluation results for each opponent
    """
    print("=" * 50)
    print("Pokemon Showdown RL Agent Evaluation")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Battles per opponent: {n_battles}")
    print(f"Format: {battle_format}")
    print()

    # Load model
    print("Loading model...")
    # Create a dummy env just to load the model
    server_config = LocalhostServerConfiguration
    dummy_env = ShowdownEnv(
        battle_format=battle_format,
        server_configuration=server_config,
        start_challenging=False,
    )
    model = PPO.load(model_path, env=dummy_env)
    print("Model loaded successfully!")
    print()

    # Evaluate against each opponent
    opponents = ["random", "max_damage", "heuristics"]
    results = []

    print("Running evaluations:")
    for opponent in opponents:
        result = await evaluate_against_opponent(
            model=model,
            opponent_type=opponent,
            n_battles=n_battles,
            battle_format=battle_format,
        )
        results.append(result)

    # Print summary
    print()
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print()
    print(f"{'Opponent':<20} {'Win Rate':<15} {'Wins/Total':<15}")
    print("-" * 50)

    for r in results:
        print(
            f"{r['opponent']:<20} "
            f"{r['win_rate']:.1%}{'':^10} "
            f"{r['wins']}/{r['total']}"
        )

    print("-" * 50)
    total_wins = sum(r["wins"] for r in results)
    total_battles = sum(r["total"] for r in results)
    overall = total_wins / total_battles if total_battles > 0 else 0
    print(f"{'Overall':<20} {overall:.1%}{'':^10} {total_wins}/{total_battles}")
    print()

    # Performance targets
    print("Performance Targets:")
    targets = {
        "random": 0.90,
        "max_damage": 0.80,
        "heuristics": 0.70,
    }

    all_passed = True
    for r in results:
        target = targets.get(r["opponent"], 0.5)
        passed = r["win_rate"] >= target
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  vs {r['opponent']}: {r['win_rate']:.1%} >= {target:.0%} [{status}]")

    print()
    if all_passed:
        print("All targets passed! Your agent is ready for real battles.")
    else:
        print("Some targets not met. Consider more training.")

    return results


def list_available_models():
    """List all available trained models."""
    print("Available models:")
    print("-" * 50)

    models = list(MODELS_DIR.glob("*.zip"))
    models.extend(MODELS_DIR.glob("**/*.zip"))

    if not models:
        print("  No models found in models/ directory")
        print("  Train a model first with: python -m src.train")
    else:
        for model_path in sorted(models):
            # Get relative path and remove .zip extension
            rel_path = model_path.relative_to(MODELS_DIR)
            print(f"  {rel_path.with_suffix('')}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pokemon Showdown RL Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (without .zip extension)"
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="Number of battles per opponent (default: 100)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen9randombattle",
        help="Battle format (default: gen9randombattle)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if not args.model:
        print("Error: --model is required")
        print("Use --list to see available models")
        return

    # Run evaluation
    asyncio.run(full_evaluation(
        model_path=args.model,
        n_battles=args.battles,
        battle_format=args.format,
    ))


if __name__ == "__main__":
    main()
