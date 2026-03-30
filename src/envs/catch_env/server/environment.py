"""
envs/catch_env/server/environment.py
--------------------------------------
Pure-Python implementation of OpenSpiel's Catch game.
No openspiel package needed — runs locally with zero extra deps.

Rules (identical to OpenSpiel):
  - 10×5 grid
  - Ball starts at row 0, random column
  - Ball falls one row per step  (rows 0 → 9)
  - Paddle starts at row 9, col 2 (centre)
  - Paddle moves ±1 col per step (or stays)
  - Terminal when ball reaches row 9
  - Reward: +1 if paddle_col == ball_col at terminal step, else 0

info_state encoding (50 floats, row-major):
  index = row * COLS + col
  1.0 at ball's (row, col)
  1.0 at paddle's (row=9, col)
  Everything else 0.0
"""

import random
import uuid
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from core.env_server import Environment
from envs.catch_env.models import (
    COLS, ROWS,
    ACTION_LEFT, ACTION_RIGHT, ACTION_STAY,
    CatchAction, CatchObservation, CatchState,
)


class CatchEnvironment(Environment):
    """
    Self-contained Catch environment.
    Identical behaviour to OpenSpiel's Catch — same grid, same reward.
    """

    def __init__(self, rows: int = ROWS, cols: int = COLS, seed: Optional[int] = None):
        self._rows = rows
        self._cols = cols
        self._rng  = random.Random(seed)

        # episode state
        self._ball_row:   int = 0
        self._ball_col:   int = 0
        self._paddle_col: int = cols // 2
        self._step:       int = 0
        self._done:       bool = False
        self._episode_id: Optional[str] = None
        self._cum_reward: float = 0.0

        # lifetime stats
        self._total_episodes: int = 0
        self._total_catches:  int = 0

    # ------------------------------------------------------------------ #
    # reset()
    # ------------------------------------------------------------------ #

    def reset(self) -> CatchObservation:
        self._ball_row   = 0
        self._ball_col   = self._rng.randint(0, self._cols - 1)
        self._paddle_col = self._cols // 2
        self._step       = 0
        self._done       = False
        self._episode_id = str(uuid.uuid4())[:8]
        self._cum_reward = 0.0
        self._total_episodes += 1
        return self._make_obs(reward=0.0, caught=False)

    # ------------------------------------------------------------------ #
    # step()
    # ------------------------------------------------------------------ #

    def step(self, action: CatchAction) -> CatchObservation:
        if self._done:
            return self._make_obs(reward=0.0, caught=False)

        # 1. Move paddle
        if action.action_id == ACTION_LEFT:
            self._paddle_col = max(0, self._paddle_col - 1)
        elif action.action_id == ACTION_RIGHT:
            self._paddle_col = min(self._cols - 1, self._paddle_col + 1)
        # ACTION_STAY → no change

        # 2. Drop ball one row
        self._ball_row += 1
        self._step     += 1

        # 3. Check terminal (ball reached bottom row)
        terminal = (self._ball_row >= self._rows - 1)
        caught   = terminal and (self._ball_col == self._paddle_col)
        reward   = 1.0 if caught else 0.0

        if terminal:
            self._done = True
            if caught:
                self._total_catches += 1

        self._cum_reward += reward
        return self._make_obs(reward=reward, caught=caught)

    # ------------------------------------------------------------------ #
    # state property
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CatchState:
        catch_rate = (
            self._total_catches / self._total_episodes
            if self._total_episodes > 0 else 0.0
        )
        return CatchState(
            episode_id=self._episode_id,
            step_count=self._step,
            game_name="catch",
            agent_player=0,
            opponent_policy="none",
            num_players=1,
            rows=self._rows,
            cols=self._cols,
            total_episodes=self._total_episodes,
            total_catches=self._total_catches,
            catch_rate=round(catch_rate, 4),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _make_obs(self, reward: float, caught: bool) -> CatchObservation:
        info_state = self._encode_info_state()
        terminal   = self._done
        return CatchObservation(
            done=terminal,
            reward=reward,
            info_state=info_state,
            legal_actions=[ACTION_LEFT, ACTION_STAY, ACTION_RIGHT],
            game_phase="terminal" if terminal else "in_progress",
            current_player_id=0,
            opponent_last_action=None,
            ball_col=self._ball_col,
            ball_row=self._ball_row,
            paddle_col=self._paddle_col,
            caught=caught,
            episode_step=self._step,
            cumulative_reward=self._cum_reward,
        )

    def _encode_info_state(self) -> list:
        """
        50-element flat list.
        Row-major: index = row * COLS + col
        Ball at its current position = 1.0
        Paddle always in row (ROWS-1) = 1.0
        """
        grid = [0.0] * (self._rows * self._cols)
        # Ball (clamp row to grid — ball starts at row 0, ends at row ROWS-1)
        ball_row = min(self._ball_row, self._rows - 1)
        grid[ball_row * self._cols + self._ball_col] = 1.0
        # Paddle (always bottom row)
        grid[(self._rows - 1) * self._cols + self._paddle_col] = 1.0
        return grid