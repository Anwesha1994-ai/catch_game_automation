"""
envs/data_cleaning_env/server/app.py
-----------------------------
FastAPI server wrapping the Catch environment.

Endpoints:
  POST /reset                → new episode
  POST /step                 → one action
  GET  /state                → episode metadata
  GET  /health               → liveness probe
  GET  /render               → ASCII art of current grid (for debugging)
  WS   /ws/visualize         → WebSocket stream: server pushes grid JSON after every step
                               (used by the HTML visualizer page)
  GET  /visualize            → serves the HTML visualizer page

Run:
  cd data_cleaning_env/src
  uvicorn envs.data_cleaning_env.server.app:app --port 8000 --reload
"""

import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

from envs.data_cleaning_env.models import CatchAction, ROWS, COLS, ACTION_NAMES, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY
# ── Simple agents for server-side demo ───────────────────────────────────────
import random
class SmartAgent:
    name = "SmartAgent"
    def act(self, obs):
        if obs['paddle_col'] < obs['ball_col']:
            return ACTION_RIGHT
        if obs['paddle_col'] > obs['ball_col']:
            return ACTION_LEFT
        return ACTION_STAY

class RandomAgent:
    name = "RandomAgent"
    def act(self, obs):
        return random.choice([ACTION_LEFT, ACTION_STAY, ACTION_RIGHT])

class AlwaysStayAgent:
    name = "AlwaysStayAgent"
    def act(self, obs):
        return ACTION_STAY

# Simple learning agent: starts random, becomes smart after a few episodes
class LearningAgent:
    name = "LearningAgent"
    def __init__(self):
        self.smart_prob = 0.0
        self.episodes = 0
    def act(self, obs):
        if random.random() < self.smart_prob:
            if obs['paddle_col'] < obs['ball_col']:
                return ACTION_RIGHT
            if obs['paddle_col'] > obs['ball_col']:
                return ACTION_LEFT
            return ACTION_STAY
        return random.choice([ACTION_LEFT, ACTION_STAY, ACTION_RIGHT])
    def learn(self, reward):
        self.episodes += 1
        if reward > 0:
            self.smart_prob = min(1.0, self.smart_prob + 0.15)
        else:
            self.smart_prob = min(1.0, self.smart_prob + 0.05)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Catch — OpenEnv", description="OpenSpiel Catch game, local pure-Python version.")

# ── Run agent endpoint ───────────────────────────────────────────────────────
from fastapi import Body
from fastapi.responses import JSONResponse

@app.post("/run_agent")
async def run_agent(
    agent_name: str = Body(..., embed=True),
    episodes: int = Body(1, embed=True),
    max_steps: int = Body(20, embed=True)
):
    # Choose agent
    agent_map = {
        "smart": SmartAgent(),
        "random": RandomAgent(),
        "stay": AlwaysStayAgent(),
        "learning": LearningAgent(),
    }
    agent = agent_map.get(agent_name)
    if agent is None:
        return JSONResponse({"error": "Unknown agent"}, status_code=400)

    stats = {"episodes": 0, "catches": 0, "history": []}
    for ep in range(episodes):
        obs = env.reset()
        d = vars(obs)
        await _broadcast({"event": "reset", **d})
        done = d.get("done", False)
        total_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            # Use dict for obs for agent
            obs_dict = d.copy()
            action_id = agent.act(obs_dict)
            action = CatchAction(action_id=action_id, game_name="catch")
            obs = env.step(action)
            d = vars(obs)
            d["action_name"] = ACTION_NAMES.get(action_id, "?")
            await _broadcast({"event": "step", **d})
            done = d.get("done", False)
            total_reward += d.get("reward", 0.0)
            steps += 1
        if hasattr(agent, "learn"):
            agent.learn(total_reward)
        stats["episodes"] += 1
        if d.get("caught", False):
            stats["catches"] += 1
        if hasattr(agent, "smart_prob"):
            stats["history"].append(round(agent.smart_prob, 2))
    stats["catch_rate"] = stats["catches"] / stats["episodes"] if stats["episodes"] else 0.0
    return stats

from envs.data_cleaning_env.server.environment import CatchEnvironment

# ── Initialise environment ────────────────────────────────────────────────────
env = CatchEnvironment()

# ── Pydantic request model ────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action_id: int = 1          # 0=LEFT 1=STAY 2=RIGHT
    game_name: str = "catch"

# Connected WebSocket visualizer clients
_ws_clients: list[WebSocket] = []

async def _broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "game": "catch", "grid": f"{ROWS}x{COLS}"}

@app.post("/reset")
async def reset():
    obs = env.reset()
    d = vars(obs)
    await _broadcast({"event": "reset", **d})
    return d

@app.post("/step")
async def step(req: StepRequest):
    action = CatchAction(action_id=req.action_id, game_name=req.game_name)
    obs = env.step(action)
    d = vars(obs)
    d["action_name"] = ACTION_NAMES.get(req.action_id, "?")
    await _broadcast({"event": "step", **d})
    return d

@app.get("/state")
def state():
    return vars(env.state)

@app.get("/render")
def render():
    """Return ASCII art of the current grid. Useful for quick debugging in the terminal."""
    rows, cols = ROWS, COLS
    ball_row  = env._ball_row
    ball_col  = env._ball_col
    paddle_col = env._paddle_col
    lines = []
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if r == ball_row and c == ball_col:
                row_str += " ● "
            elif r == rows - 1 and c == paddle_col:
                row_str += "═══"
            else:
                row_str += " · "
        lines.append(row_str)
    lines.append(f"\nBall: row={ball_row} col={ball_col}  |  Paddle: col={paddle_col}")
    lines.append(f"Step: {env._step}  Done: {env._done}")
    return {"ascii": "\n".join(lines)}

# ── WebSocket for live visualizer ─────────────────────────────────────────────

@app.websocket("/ws/visualize")
async def ws_visualize(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    # Send current state immediately on connect
    obs = env._make_obs(reward=0.0, caught=False)
    await websocket.send_text(json.dumps({"event": "init", **vars(obs)}))
    try:
        while True:
            await asyncio.sleep(10)   # keep alive
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)

# ── HTML Visualizer page ──────────────────────────────────────────────────────

@app.get("/visualize", response_class=HTMLResponse)
def visualize():
    html_path = os.path.join(os.path.dirname(__file__), "visualizer.html")
    return FileResponse(html_path, media_type="text/html")