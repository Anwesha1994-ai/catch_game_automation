from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar
import requests
from core.env_server import Action, Observation, State

ActionT = TypeVar("ActionT", bound=Action)
ObsT    = TypeVar("ObsT",    bound=Observation)

@dataclass
class StepResult:
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]

class HTTPEnvClient(ABC, Generic[ActionT, ObsT]):
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def reset(self) -> StepResult:
        r = requests.post(f"{self.base_url}/reset", timeout=self.timeout)
        r.raise_for_status()
        return self._parse_result(r.json())

    def step(self, action: ActionT) -> StepResult:
        r = requests.post(f"{self.base_url}/step", json=self._step_payload(action), timeout=self.timeout)
        r.raise_for_status()
        return self._parse_result(r.json())

    def state(self) -> State:
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return self._parse_state(r.json())

    @abstractmethod
    def _step_payload(self, action: ActionT) -> dict: ...
    @abstractmethod
    def _parse_result(self, payload: dict) -> StepResult: ...
    @abstractmethod
    def _parse_state(self, payload: dict) -> State: ...