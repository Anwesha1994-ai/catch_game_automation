"""
envs/data_cleaning_env/models.py
------------------------
Typed contracts that EXACTLY match the OpenSpiel Catch environment API.

Catch grid: 10 rows × 5 cols  →  info_state = 50 floats (flattened)
Actions:
    0 = LEFT
    1 = STAY
    2 = RIGHT
Reward:
    +1.0  if ball caught on final step
     0.0  if ball missed on final step
     0.0  on every non-terminal step
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.env_server import Action, Observation, State

ROWS = 10
COLS = 5
ACTION_LEFT  = 0
ACTION_STAY  = 1
ACTION_RIGHT = 2
ACTION_NAMES = {0: "LEFT ←", 1: "STAY  ·", 2: "RIGHT →"}

@dataclass
class CatchAction(Action):
    """Exactly mirrors OpenSpiel's OpenSpielAction for Catch."""
    action_id: int = ACTION_STAY          # 0 LEFT | 1 STAY | 2 RIGHT
    game_name: str = "catch"
    game_params: Dict[str, Any] = field(default_factory=dict)
    metadata:   Dict[str, Any] = field(default_factory=dict)

@dataclass
class CatchObservation(Observation):
    """
    Mirrors OpenSpiel's OpenSpielObservation.
    info_state is the 50-element flattened grid (row-major):
      index = row*COLS + col
      1.0 → ball position  (in its current row)
      1.0 → paddle position (always in row 9, the bottom row)
    """
    done:          bool  = False
    reward:        float = 0.0

    # ── Core OpenSpiel fields ──────────────────────────────────────────
    info_state:          List[float] = field(default_factory=list)
    legal_actions:       List[int]   = field(default_factory=lambda: [0,1,2])
    game_phase:          str         = "in_progress"   # "in_progress" | "terminal"
    current_player_id:   int         = 0
    opponent_last_action: Optional[int] = None

    # ── Convenience extras (not in OpenSpiel, helpful for learning) ────
    ball_col:    int   = 0
    ball_row:    int   = 0
    paddle_col:  int   = 2
    caught:      bool  = False          # True if ball was caught (terminal)
    episode_step: int  = 0
    cumulative_reward: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CatchState(State):
    """Episode-level metadata — returned by GET /state."""
    episode_id:   Optional[str] = None
    step_count:   int   = 0
    game_name:    str   = "catch"
    agent_player: int   = 0
    opponent_policy: str = "none"       # Catch is single-player
    num_players:  int   = 1
    rows:         int   = ROWS
    cols:         int   = COLS
    total_episodes: int = 0
    total_catches:  int = 0
    catch_rate:     float = 0.0