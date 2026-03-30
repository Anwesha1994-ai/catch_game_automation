from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from fastapi import FastAPI

@dataclass
class Action:
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Observation:
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class State:
    episode_id: Optional[str] = None
    step_count: int = 0

class Environment(ABC):
    @abstractmethod
    def reset(self) -> Observation: ...
    @abstractmethod
    def step(self, action: Action) -> Observation: ...
    @property
    @abstractmethod
    def state(self) -> State: ...

def create_fastapi_app(env: Environment) -> FastAPI:
    app = FastAPI(title="OpenEnv — Catch")

    @app.post("/reset")
    def reset():
        return vars(env.reset())

    @app.post("/step")
    def step(payload: Dict[str, Any]):
        from envs.data_cleaning_env.models import CatchAction
        action = CatchAction(action_id=int(payload.get("action_id", 1)),
                             game_name=payload.get("game_name", "catch"))
        return vars(env.step(action))

    @app.get("/state")
    def state():
        return vars(env.state)

    @app.get("/health")
    def health():
        return {"status": "ok", "game": "catch"}

    return app