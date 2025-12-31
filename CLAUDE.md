# Pokemon Showdown AI - Project Status

> **REMINDER TO CLAUDE**: Update this file after every user command. Track training progress, issues encountered, and next steps.

---

## Current Status: READY TO TRAIN

- [x] Project structure created
- [x] Dependencies defined (requirements.txt)
- [x] Environment wrapper implemented
- [x] Training pipeline implemented
- [x] Evaluation script implemented
- [x] Play interface implemented
- [x] Dependencies installed (poke-env, stable-baselines3, torch, tensorboard)
- [x] Local Showdown server installed
- [x] README.md created
- [x] .gitignore created
- [ ] Stage 1 training started
- [ ] Stage 2 training started
- [ ] Stage 3 self-play training started

---

## Training Progress

| Stage | Status | Timesteps | Win Rate vs Target |
|-------|--------|-----------|-------------------|
| Stage 1 (vs MaxDamage) | Not started | 0 / 50,000 | - |
| Stage 2 (vs Mixed) | Not started | 0 / 100,000 | - |
| Stage 3 (Self-play) | Not started | 0 / 500,000+ | - |

**Latest Model**: None yet

---

## How Training Works

### One-Time vs Continuous Training

**No, you don't run training just once!** The training is designed to be:

1. **Resumable** - You can stop and continue later
2. **Incremental** - Each run improves the model further
3. **Unlimited** - Self-play can run as long as you want

### The Three Stages

| Stage | Purpose | Default Timesteps | When to Move On |
|-------|---------|-------------------|-----------------|
| **Stage 1** | Learn basics (type effectiveness, switching) | 50,000 | >70% vs MaxDamage |
| **Stage 2** | Generalize to different opponents | 100,000 | >60% vs all baselines |
| **Stage 3** | Self-play mastery | 500,000+ | Never "done" - keeps improving |

### Continuing Training

To train longer, you have two options:

**Option 1: Increase timesteps**
```bash
# Train for 200k steps instead of 50k
python -m src.train --stage 1 --timesteps 200000
```

**Option 2: Resume from a saved model**
```bash
# Continue training an existing model
python -m src.train --stage 1 --model models/stage1_final_TIMESTAMP --timesteps 50000
```

**Option 3: Loop training (recommended for self-play)**
```bash
# Train self-play in a loop, 100k steps at a time
while true; do
    python -m src.train --stage 3 --timesteps 100000
    echo "Completed a training cycle, continuing..."
    sleep 5
done
```

### How Long to Train?

| Hardware | Stage 1 (50k) | Stage 2 (100k) | Stage 3 (500k) |
|----------|---------------|----------------|----------------|
| CPU only | ~2-4 hours | ~4-8 hours | ~20-40 hours |
| GPU (RTX 3060+) | ~30-60 min | ~1-2 hours | ~5-10 hours |

**More training = better performance**, but diminishing returns after ~1M timesteps for self-play.

---

## Quick Reference Commands

### Setup (run once)
```bash
pip install -r requirements.txt
./setup_server.sh
```

### Start server (before any training/playing)
```bash
./start_server.sh
```

### Training
```bash
# Stage 1 - basics
python -m src.train --stage 1 --timesteps 50000

# Stage 2 - generalization
python -m src.train --stage 2 --timesteps 100000

# Stage 3 - self-play (run multiple times!)
python -m src.train --stage 3 --timesteps 100000

# Continue from existing model
python -m src.train --stage 3 --model models/selfplay_final_TIMESTAMP --timesteps 100000
```

### Evaluation
```bash
python -m src.evaluate --model models/MODEL_NAME --battles 100
```

### Play
```bash
# Local
python -m src.play --model models/MODEL_NAME --local

# Public ladder
python -m src.play --model models/MODEL_NAME --username BOT --password PASS --ladder 10
```

### Monitor training
```bash
tensorboard --logdir logs/
```

---

## Issues & Notes

- **2024-12-30**: Initial `pip install` failed due to /tmp being RAM-based (3.8GB limit). Fixed by using `TMPDIR=/path/to/disk pip install`. PyTorch + CUDA libs are ~4GB.

---

## Next Steps

1. ~~Install Python dependencies~~ DONE
2. ~~Install local Showdown server~~ DONE
3. Start the server: `./start_server.sh`
4. Begin Stage 1 training: `python -m src.train --stage 1`

---

*Last updated: 2024-12-30 - Setup complete, ready to train*
