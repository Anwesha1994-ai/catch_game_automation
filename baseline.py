"""
baseline.py
-----------
Baseline script for the Catch environment.
Runs 3 agents (Smart, Random, Always-Stay) and prints scores.

Usage:
    # Terminal 1 — start server:
    cd src && uvicorn envs.catch_env.server.app:app --port 8000

    # Terminal 2 — run baselines:
    python baseline.py

    # Or run with ASCII visualisation (slower):
    python baseline.py --render
"""

import argparse
import random
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from envs.catch_env.client import CatchEnv
from envs.catch_env.models import CatchAction, ROWS, COLS, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY

# ── Learning Agent ───────────────────────────────────────────────────────────

class LearningAgent:
    """
    Starts as a random agent, but gradually learns to act like SmartAgent.
    After each episode, it updates its policy to move the paddle toward the ball more often.
    """
    name = "LearningAgent (random→smart)"
    def __init__(self):
        self.smart_prob = 0.0  # Probability of using smart action
        self.episodes = 0
        self.history = []

    def act(self, obs) -> CatchAction:
        # With probability smart_prob, act like SmartAgent
        if random.random() < self.smart_prob:
            if obs.paddle_col < obs.ball_col:
                return CatchAction(action_id=ACTION_RIGHT)
            if obs.paddle_col > obs.ball_col:
                return CatchAction(action_id=ACTION_LEFT)
            return CatchAction(action_id=ACTION_STAY)
        # Otherwise, act randomly
        return CatchAction(action_id=random.choice(obs.legal_actions))

    def learn(self, reward):
        # After each episode, increase smart_prob if reward was positive
        self.episodes += 1
        if reward > 0:
            self.smart_prob = min(1.0, self.smart_prob + 0.15)
        else:
            self.smart_prob = min(1.0, self.smart_prob + 0.05)
        self.history.append(self.smart_prob)

SEED = 42
random.seed(SEED)


# ── Agents ─────────────────────────────────────────────────────────────────────

class SmartAgent:
    """Optimal: always moves paddle toward the ball column."""
    name = "SmartAgent  (optimal)"
    def act(self, obs) -> CatchAction:
        if obs.paddle_col < obs.ball_col:   return CatchAction(action_id=ACTION_RIGHT)
        if obs.paddle_col > obs.ball_col:   return CatchAction(action_id=ACTION_LEFT)
        return CatchAction(action_id=ACTION_STAY)

class RandomAgent:
    """Random: picks uniformly from legal actions."""
    name = "RandomAgent (baseline)"
    def act(self, obs) -> CatchAction:
        return CatchAction(action_id=random.choice(obs.legal_actions))

class AlwaysStayAgent:
    """Never moves — catches only if ball lands on starting col 2."""
    name = "StayAgent   (lower bound)"
    def act(self, obs) -> CatchAction:
        return CatchAction(action_id=ACTION_STAY)


# ── ASCII render ───────────────────────────────────────────────────────────────

def ascii_grid(obs) -> str:
    lines = ["┌" + "─" * (COLS * 3) + "┐"]
    for r in range(ROWS):
        row = "│"
        for c in range(COLS):
            if r == obs.ball_row and c == obs.ball_col:
                row += " ● "
            elif r == ROWS - 1 and c == obs.paddle_col:
                row += "═══"
            else:
                row += " · "
        row += "│"
        lines.append(row)
    lines.append("└" + "─" * (COLS * 3) + "┘")
    return "\n".join(lines)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env, agent, render=False) -> float:
    result = env.reset()
    obs = result.observation
    total_reward = 0.0

    while not obs.done:
        if render:
            print("\033[H\033[J", end="")   # clear terminal
            print(f"Agent: {agent.name}")
            print(ascii_grid(obs))
            print(f"Step {obs.episode_step}  |  Ball:({obs.ball_row},{obs.ball_col})  Paddle:{obs.paddle_col}")
            time.sleep(0.18)

        action = agent.act(obs)
        result = env.step(action)
        obs    = result.observation
        total_reward += result.reward

    if hasattr(agent, "learn"):
        agent.learn(total_reward)

    if render:
        print("\033[H\033[J", end="")
        print(ascii_grid(obs))
        print("🎉 CAUGHT!" if obs.caught else "✗ Missed")
        time.sleep(0.6)

    return total_reward


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(env, agent, n=20, render=False):
    rewards = [run_episode(env, agent, render=render) for _ in range(n)]
    catch_rate = sum(r > 0 for r in rewards) / n
    return {"catch_rate": round(catch_rate, 4), "mean_reward": round(sum(rewards)/n, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render",   action="store_true", help="Show ASCII grid in terminal")
    args = p.parse_args()

    env    = CatchEnv(base_url=args.base_url)
    agents = [SmartAgent(), RandomAgent(), AlwaysStayAgent(), LearningAgent()]

    print("=" * 58)
    print("  Catch — OpenEnv Baselines")
    print(f"  Grid: {ROWS}×{COLS}  |  Episodes: {args.episodes}  |  Seed: {SEED}")
    print("=" * 58)

    for agent in agents:
        stats = evaluate(env, agent, n=args.episodes, render=args.render and isinstance(agent, SmartAgent))
        bar   = "█" * int(stats["catch_rate"] * 40)
        print(f"\n  {agent.name}")
        print(f"  [{bar:<40}] {stats['catch_rate']*100:.1f}% catch rate")
        print(f"  mean_reward = {stats['mean_reward']:.4f}")
        if hasattr(agent, "history"):
            print(f"  smart_prob history: {[round(p,2) for p in agent.history]}")

    print("\n" + "=" * 58)
    print("  Open http://localhost:8000/visualize for live demo")
    print("=" * 58)


if __name__ == "__main__":
    main()