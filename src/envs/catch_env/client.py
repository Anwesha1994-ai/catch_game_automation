"""
envs/catch_env/client.py
-------------------------
Typed HTTP client — mirrors OpenSpiel's OpenSpielEnv exactly.

Usage:
    from envs.catch_env.client import CatchEnv
    from envs.catch_env.models import CatchAction

    env = CatchEnv(base_url="http://localhost:8000")
    result = env.reset()
    print(result.observation.info_state)   # 50 floats

    result = env.step(CatchAction(action_id=2))  # RIGHT
    print(result.reward, result.done)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from core.http_env_client import HTTPEnvClient, StepResult
from envs.catch_env.models import CatchAction, CatchObservation, CatchState


class CatchEnv(HTTPEnvClient[CatchAction, CatchObservation]):

    def _step_payload(self, action: CatchAction) -> dict:
        return {"action_id": action.action_id, "game_name": action.game_name}

    def _parse_result(self, p: dict) -> StepResult:
        obs = CatchObservation(
            done=p.get("done", False),
            reward=p.get("reward", 0.0),
            info_state=p.get("info_state", []),
            legal_actions=p.get("legal_actions", [0,1,2]),
            game_phase=p.get("game_phase", "in_progress"),
            current_player_id=p.get("current_player_id", 0),
            opponent_last_action=p.get("opponent_last_action"),
            ball_col=p.get("ball_col", 0),
            ball_row=p.get("ball_row", 0),
            paddle_col=p.get("paddle_col", 2),
            caught=p.get("caught", False),
            episode_step=p.get("episode_step", 0),
            cumulative_reward=p.get("cumulative_reward", 0.0),
        )
        return StepResult(observation=obs, reward=obs.reward, done=obs.done,
                          info={"game_phase": obs.game_phase, "caught": obs.caught})

    def _parse_state(self, p: dict) -> CatchState:
        return CatchState(
            episode_id=p.get("episode_id"),
            step_count=p.get("step_count", 0),
            game_name=p.get("game_name", "catch"),
            total_episodes=p.get("total_episodes", 0),
            total_catches=p.get("total_catches", 0),
            catch_rate=p.get("catch_rate", 0.0),
        )