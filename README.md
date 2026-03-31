# 🎓 **AIML Project**

**Name:** Anwesha Nandi  
**Reg No.:** 25BCE11262  
**Branch:** CSE (CORE)  
**Year:** 1st Year B.Tech   
---

# 🏓 Catch Game Learning Agent OpenEnv

An **OpenEnv** environment where an AI agent learns to play the Catch game—a simple grid-based game where the agent controls a paddle to catch a falling ball. This project is designed for reinforcement learning research and benchmarking, providing a clean API, baseline agents, and a FastAPI server for easy experimentation.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---


## 🌍 What is this environment?

The Catch environment simulates a classic reinforcement learning task:

- **Grid**: 10 rows × 5 columns
- **Agent**: Controls a paddle at the bottom row
- **Goal**: Move the paddle left or right to catch a falling ball
- **Reward**: +1 for catching the ball, 0 otherwise

This environment is ideal for testing RL algorithms, curriculum learning, and agent benchmarking. It is fully compatible with OpenEnv standards and includes a typed HTTP client, server, and several baseline agents.

# 🏓 Catch Environment OpenEnv

An **OpenEnv** environment where an AI agent learns to play the Catch game, a grid-based simulation where the agent controls a paddle to catch a falling ball. The environment is designed for reinforcement learning experiments and includes:

- **Typed HTTP Client**: For interacting with the environment server.
- **FastAPI Server**: Hosting the Catch environment simulation.
- **Baseline Agents**: Smart, Random, and Learning agents for benchmarking.

---

## 🌍 What is this environment?

The Catch environment simulates a simple grid-based game:
- **Grid**: 10 rows × 5 columns.
- **Objective**: Move the paddle to catch the falling ball.
- **Reward**: +1 for catching the ball, 0 otherwise.

This environment is compatible with OpenEnv standards and can be used for reinforcement learning experiments.

---

## 🎯 Three Agents

| Agent          | Description                          |
|----------------|--------------------------------------|
| SmartAgent     | Always moves toward the ball column. |
| RandomAgent    | Picks actions randomly.              |
| LearningAgent  | Learns to act like SmartAgent over time. |

---

## 📐 Action Space

Each step, the agent sends one action:

| Action ID | Action |
|-----------|--------|
| 0         | LEFT   |
| 1         | STAY   |
| 2         | RIGHT  |

---

## 👁️ Observation Space

Each `step()` and `reset()` returns:

```python
{
  "done": false,
  "reward": 0.0,
  "ball_row": 5,
  "ball_col": 2,
  "paddle_col": 3,
  "episode_step": 10,
  "caught": false
}
```

---


## 🐳 Docker

```bash
# Build
docker build -t catch-env .

# Run
docker run -p 7860:7860 catch-env
```

---

## 🚀 Quick Start

### 1. Setup Env
```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the baselines

```bash
python baseline.py --base-url http://localhost:7860
```

---

## 📁 Project Structure

```
catch_env/
├── src/
│   ├── core/
│   │   ├── env_server.py          ← Base classes (Environment, Action, Observation)
│   │   └── http_env_client.py     ← Base HTTP client
│   └── envs/
│       └── catch_env/
│           ├── models.py          ← Typed Action, Observation, State
│           ├── tasks.py           ← Tasks + agent graders
│           ├── client.py          ← CatchEnv client
│           └── server/
│               ├── environment.py ← Core simulation logic
│               └── app.py         ← FastAPI server
├── baseline.py                    ← Baseline agents
├── openenv.yaml                   ← Environment spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Baseline Scores (seed=42)

| Agent         | Catch Rate |
|---------------|------------|
| SmartAgent    | ~100%      |
| RandomAgent   | ~20%       |
| LearningAgent | Improves over time |

---
