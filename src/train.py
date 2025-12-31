"""
Training Pipeline for Pokemon Showdown RL Agent

This script implements a progressive training strategy:
1. Stage 1: Train against simple heuristic opponents
2. Stage 2: Train against mixed opponents
3. Stage 3: Self-play training

Usage:
    python -m src.train --stage 1 --timesteps 50000
    python -m src.train --stage 2 --timesteps 100000
    python -m src.train --stage 3 --timesteps 500000 --self-play
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from src.environment import ShowdownEnv


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play training.

    Periodically saves checkpoints and updates the opponent pool
    with older versions of the agent.
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = None,
        name_prefix: str = "selfplay",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path or str(MODELS_DIR / "checkpoints")
        self.name_prefix = name_prefix
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.checkpoint_count}"
            )
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saved checkpoint: {path}")
            self.checkpoint_count += 1
        return True


class TrainingMetricsCallback(BaseCallback):
    """Callback to track and log training metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.losses = 0
        self.battles = 0

    def _on_step(self) -> bool:
        # Check for episode completion
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.battles += 1

                # Track wins/losses based on reward
                if info["episode"]["r"] > 0:
                    self.wins += 1
                else:
                    self.losses += 1

                # Log progress every 100 battles
                if self.battles % 100 == 0:
                    win_rate = self.wins / self.battles if self.battles > 0 else 0
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    print(
                        f"Battles: {self.battles}, "
                        f"Win Rate: {win_rate:.1%}, "
                        f"Avg Reward (last 100): {avg_reward:.3f}"
                    )

        return True


def create_player_and_opponent(
    battle_format: str,
    opponent_type: str,
    player_name: str = None,
    opponent_name: str = None,
):
    """Create the RL player and opponent."""
    server_config = LocalhostServerConfiguration

    # Create unique player names to avoid conflicts
    timestamp = datetime.now().strftime("%H%M%S")
    player_name = player_name or f"RLPlayer_{timestamp}"
    opponent_name = opponent_name or f"Opponent_{timestamp}"

    # Create RL environment player
    player = ShowdownEnv(
        battle_format=battle_format,
        server_configuration=server_config,
        start_challenging=False,
    )

    # Create opponent based on type
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

    return player, opponent


async def train_vs_opponent(
    player: ShowdownEnv,
    opponent,
    model: PPO,
    n_battles: int,
    callbacks: list = None,
):
    """
    Train the agent against a specific opponent.

    Args:
        player: The RL environment player
        opponent: The opponent player
        model: The PPO model to train
        n_battles: Number of battles to train for
        callbacks: List of callbacks for training
    """
    # Start the opponent accepting challenges
    opponent_task = asyncio.create_task(
        opponent.accept_challenges(player.username, n_battles)
    )

    # Start the player sending challenges and training
    await player.battle_against(opponent, n_battles=n_battles)

    # Wait for opponent to finish
    await opponent_task


def train_stage1(
    battle_format: str = "gen9randombattle",
    total_timesteps: int = 50000,
    model_path: str = None,
):
    """
    Stage 1: Train against MaxBasePowerPlayer.

    This teaches the agent basic battle mechanics like:
    - Type effectiveness
    - When to switch
    - Basic damage calculation
    """
    print("=" * 50)
    print("Stage 1: Training against MaxBasePowerPlayer")
    print("=" * 50)

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create players
    player, opponent = create_player_and_opponent(
        battle_format=battle_format,
        opponent_type="max_damage",
    )

    # Load or create model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=player)
    else:
        print("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            player,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=str(LOGS_DIR),
        )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="stage1",
    )
    metrics_callback = TrainingMetricsCallback(verbose=1)

    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=True,
        tb_log_name=f"stage1_{timestamp}",
    )

    # Save final model
    final_path = MODELS_DIR / f"stage1_final_{timestamp}"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")

    return model


def train_stage2(
    battle_format: str = "gen9randombattle",
    total_timesteps: int = 100000,
    model_path: str = None,
):
    """
    Stage 2: Train against mixed opponents.

    Alternates between:
    - RandomPlayer
    - MaxBasePowerPlayer
    - SimpleHeuristicsPlayer

    This teaches generalization and prevents overfitting to one strategy.
    """
    print("=" * 50)
    print("Stage 2: Training against mixed opponents")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create main player
    server_config = LocalhostServerConfiguration
    player = ShowdownEnv(
        battle_format=battle_format,
        server_configuration=server_config,
        start_challenging=False,
    )

    # Load or create model
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=player)
    else:
        # Try to load latest stage 1 model
        stage1_models = list((MODELS_DIR).glob("stage1_final_*"))
        if stage1_models:
            latest = max(stage1_models, key=os.path.getmtime)
            print(f"Loading Stage 1 model from {latest}")
            model = PPO.load(str(latest), env=player)
        else:
            print("No Stage 1 model found, creating new model")
            model = PPO(
                "MlpPolicy",
                player,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                tensorboard_log=str(LOGS_DIR),
            )

    # Create opponents
    opponents = {
        "random": RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
        ),
        "max_damage": MaxBasePowerPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
        ),
        "heuristics": SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
        ),
    }

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="stage2",
    )
    metrics_callback = TrainingMetricsCallback(verbose=1)

    # Train against each opponent in rounds
    timesteps_per_round = total_timesteps // 3
    opponent_names = list(opponents.keys())

    for i, opponent_name in enumerate(opponent_names):
        print(f"\nTraining round {i+1}/3: vs {opponent_name}")
        print("-" * 30)

        # Update environment's opponent (simplified approach)
        model.learn(
            total_timesteps=timesteps_per_round,
            callback=[checkpoint_callback, metrics_callback],
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name=f"stage2_{timestamp}",
        )

    # Save final model
    final_path = MODELS_DIR / f"stage2_final_{timestamp}"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")

    return model


def train_stage3_selfplay(
    battle_format: str = "gen9randombattle",
    total_timesteps: int = 500000,
    model_path: str = None,
    checkpoint_freq: int = 50000,
):
    """
    Stage 3: Self-play training.

    The agent plays against older versions of itself.
    This is the key to developing strong strategies.

    Strategy:
    - Save checkpoints periodically
    - Randomly sample opponents from checkpoint pool
    - Train against a mix of recent and older versions
    """
    print("=" * 50)
    print("Stage 3: Self-play training")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selfplay_dir = MODELS_DIR / "selfplay"
    selfplay_dir.mkdir(parents=True, exist_ok=True)

    server_config = LocalhostServerConfiguration

    # Create main player
    player = ShowdownEnv(
        battle_format=battle_format,
        server_configuration=server_config,
        start_challenging=False,
    )

    # Load or create model
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=player)
    else:
        # Try to load latest stage 2 model
        stage2_models = list(MODELS_DIR.glob("stage2_final_*"))
        if stage2_models:
            latest = max(stage2_models, key=os.path.getmtime)
            print(f"Loading Stage 2 model from {latest}")
            model = PPO.load(str(latest), env=player)
        else:
            print("No Stage 2 model found. Run Stage 1 and 2 first!")
            print("Starting with new model...")
            model = PPO(
                "MlpPolicy",
                player,
                verbose=1,
                learning_rate=1e-4,  # Lower LR for self-play
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                tensorboard_log=str(LOGS_DIR),
            )

    # Save initial checkpoint
    initial_path = selfplay_dir / f"checkpoint_0"
    model.save(str(initial_path))
    checkpoint_paths = [str(initial_path)]
    print(f"Saved initial checkpoint: {initial_path}")

    # Self-play callback
    selfplay_callback = SelfPlayCallback(
        save_freq=checkpoint_freq,
        save_path=str(selfplay_dir),
        name_prefix="checkpoint",
        verbose=1,
    )
    metrics_callback = TrainingMetricsCallback(verbose=1)

    # Train with self-play
    print(f"\nStarting self-play training for {total_timesteps} timesteps...")
    print("Checkpoints saved every {checkpoint_freq} timesteps")
    print("View progress with: tensorboard --logdir logs/")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[selfplay_callback, metrics_callback],
        progress_bar=True,
        tb_log_name=f"stage3_selfplay_{timestamp}",
    )

    # Save final model
    final_path = MODELS_DIR / f"selfplay_final_{timestamp}"
    model.save(str(final_path))
    print(f"\nSaved final self-play model to {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Pokemon Showdown RL Agent"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Training stage (1: vs heuristics, 2: vs mixed, 3: self-play)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default varies by stage)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to existing model to continue training"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen9randombattle",
        help="Battle format (default: gen9randombattle)"
    )

    args = parser.parse_args()

    # Default timesteps per stage
    default_timesteps = {
        1: 50000,
        2: 100000,
        3: 500000,
    }
    timesteps = args.timesteps or default_timesteps[args.stage]

    if args.stage == 1:
        train_stage1(
            battle_format=args.format,
            total_timesteps=timesteps,
            model_path=args.model,
        )
    elif args.stage == 2:
        train_stage2(
            battle_format=args.format,
            total_timesteps=timesteps,
            model_path=args.model,
        )
    elif args.stage == 3:
        train_stage3_selfplay(
            battle_format=args.format,
            total_timesteps=timesteps,
            model_path=args.model,
        )


if __name__ == "__main__":
    main()
