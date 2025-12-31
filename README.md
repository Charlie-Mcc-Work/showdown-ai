# Pokemon Showdown AI

A self-play reinforcement learning AI for Pokemon Showdown's Gen 9 Random Battle format.

## Overview

This project trains an AI agent to play Pokemon battles using:
- **poke-env**: Python interface to Pokemon Showdown
- **Stable-Baselines3**: PPO reinforcement learning algorithm
- **Self-play**: Agent improves by playing against older versions of itself

## Requirements

- Python 3.10+
- Node.js 18+ (for local Showdown server)
- ~5GB disk space (for dependencies)
- GPU recommended but not required

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set up local Pokemon Showdown server
./setup_server.sh
```

## Usage

### 1. Start the Server

Before training or playing, start the local Showdown server in a separate terminal:

```bash
./start_server.sh
```

Leave this running in the background.

### 2. Train the AI

Training happens in three stages:

```bash
# Stage 1: Learn basics against simple opponent (~50k steps)
python -m src.train --stage 1

# Stage 2: Generalize against varied opponents (~100k steps)
python -m src.train --stage 2

# Stage 3: Self-play - this is where real improvement happens
python -m src.train --stage 3
```

#### Training Longer

The AI improves with more training. Options:

```bash
# Increase timesteps
python -m src.train --stage 3 --timesteps 500000

# Resume from saved model
python -m src.train --stage 3 --model models/selfplay_final_TIMESTAMP

# Loop training overnight
while true; do
    python -m src.train --stage 3 --timesteps 100000
    sleep 5
done
```

#### Monitor Progress

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### 3. Evaluate Performance

Test your trained model against baseline opponents:

```bash
python -m src.evaluate --model models/MODEL_NAME --battles 100
```

Performance targets:
- vs RandomPlayer: >90% win rate
- vs MaxDamagePlayer: >80% win rate
- vs SimpleHeuristics: >70% win rate

### 4. Play Real Games

#### On Local Server
```bash
python -m src.play --model models/MODEL_NAME --local
```

#### On Public Pokemon Showdown
```bash
# Ladder matches
python -m src.play --model models/MODEL_NAME --username YOUR_USERNAME --password YOUR_PASSWORD --ladder 10

# Challenge specific player
python -m src.play --model models/MODEL_NAME --username YOUR_USERNAME --password YOUR_PASSWORD --challenge OPPONENT_NAME
```

## Project Structure

```
showdown/
├── src/
│   ├── environment.py   # Battle state embedding and rewards
│   ├── train.py         # Training pipeline with self-play
│   ├── evaluate.py      # Evaluation against baselines
│   └── play.py          # Real-time play interface
├── models/              # Saved model checkpoints
├── logs/                # TensorBoard training logs
├── pokemon-showdown/    # Local Showdown server (created by setup)
├── requirements.txt
├── setup_server.sh
└── start_server.sh
```

## How It Works

### State Representation

The AI sees battles as a 76-dimensional vector:
- Active Pokemon: HP, types, status, stat boosts
- Opponent's active Pokemon: HP, types, status, stat boosts
- Available moves: base power and type effectiveness
- Team HP percentages

### Reward Function

- Win: +1.0
- Loss: -1.0
- Knockout opponent Pokemon: +0.15
- Damage dealt/received: scaled rewards

### Self-Play Training

1. Agent plays against older versions of itself
2. Checkpoints saved periodically
3. Creates an "arms race" of improving strategies
4. Prevents forgetting through diverse opponents

## Training Time Estimates

| Hardware | Stage 1 (50k) | Stage 2 (100k) | Stage 3 (500k) |
|----------|---------------|----------------|----------------|
| CPU only | 2-4 hours | 4-8 hours | 20-40 hours |
| GPU | 30-60 min | 1-2 hours | 5-10 hours |

## Troubleshooting

### "No space left on device" during pip install
Your /tmp is RAM-based. Use disk instead:
```bash
mkdir -p .tmp && TMPDIR=.tmp pip install -r requirements.txt
```

### Server connection errors
Make sure the local server is running:
```bash
./start_server.sh
```

### Training is slow
- Use GPU if available (CUDA)
- Reduce `n_steps` in train.py for faster iterations
- Start with smaller timesteps to verify setup works

## License

MIT
