# 🧹 Data Cleaning OpenEnv

An **OpenEnv** environment where an AI agent learns to clean messy real-world datasets,
one operation at a time.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🌍 What is this environment?

Data quality is a billion-dollar problem.  Before any ML model can train on real data,
engineers spend 60–80% of their time cleaning it.  This environment simulates that workbench.

The agent receives a **messy Pandas DataFrame** and must apply cleaning operations step by step
to bring data quality above a threshold.  Every action returns a **partial reward** proportional
to the improvement in data quality — so the agent gets a learning signal on every step, not
just at the end.

---

## 🎯 Three Tasks (Easy → Medium → Hard)

| Task ID | Difficulty | Problems | Target Quality | Max Steps |
|---------|-----------|----------|---------------|-----------|
| `easy_missing_values` | 🟢 Easy | Missing values only | 0.90 | 15 |
| `medium_type_and_string_errors` | 🟡 Medium | Missing + type errors + inconsistent strings | 0.85 | 20 |
| `hard_full_pipeline` | 🔴 Hard | Missing + outliers + strings + whitespace + duplicates | 0.90 | 30 |

---

## 📐 Action Space

Each step the agent sends **one** cleaning operation:

```json
{
  "operation": "fill_missing_mean",
  "column": "Age",
  "params": {}
}
```

**Available operations:**

| Operation | What it does | Needs `column`? | Needs `params`? |
|-----------|-------------|----------------|----------------|
| `fill_missing_mean` | Fill NaN with column mean | ✅ | ❌ |
| `fill_missing_mode` | Fill NaN with column mode | ✅ | ❌ |
| `fill_missing_value` | Fill NaN with fixed value | ✅ | `{"value": X}` |
| `drop_rows_with_na` | Drop rows with NaN in column | ✅ | ❌ |
| `cast_column` | Change dtype | ✅ | `{"dtype": "int64"}` |
| `strip_whitespace` | Remove leading/trailing spaces | ✅ | ❌ |
| `normalize_case` | Standardise string case | ✅ | `{"mode": "title"}` |
| `replace_value` | Replace bad value with correct one | ✅ | `{"old": X, "new": Y}` |
| `clip_outliers` | Clip values to [mean ± k*std] | ✅ | `{"k": 2.0}` |
| `drop_column` | Remove a column | ✅ | ❌ |
| `rename_column` | Rename a column | ✅ | `{"new_name": "..."}` |
| `submit` | Signal done, trigger grader | ❌ | ❌ |

---

## 👁️ Observation Space

Each `step()` and `reset()` returns:

```python
{
  "done": false,
  "reward": 0.12,                 # reward for THIS step
  "cumulative_reward": 0.35,

  # Dataset statistics
  "num_rows": 60,
  "num_cols": 5,
  "column_names": ["EmployeeID", "Name", "Department", "Age", "Salary"],
  "dtypes": {"Age": "float64", "Salary": "float64", ...},
  "missing_counts": {"Age": 12, "Salary": 9},
  "total_missing": 21,
  "duplicate_rows": 0,

  # Quality scores (0.0 – 1.0) ← partial progress signals
  "completeness_score": 0.65,
  "consistency_score":  1.0,
  "uniqueness_score":   1.0,
  "overall_quality":    0.81,

  # Feedback on last action
  "last_action": "fill_missing_mean",
  "last_action_result": "success",
  "error_message": "",

  # Task context
  "task_id": "easy_missing_values",
  "target_quality": 0.90
}
```

---

## 🏆 Reward Function

```
reward = overall_quality_after - overall_quality_before

overall_quality = 0.4 * completeness
               + 0.3 * consistency
               + 0.3 * uniqueness
```

- **Positive reward** → the action improved data quality
- **-0.02** → no-op action (nothing changed)
- **-0.05** → invalid action (unknown operation, wrong column, etc.)

This gives the agent a learning signal on **every step**, not just at the end.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
cd src
TASK_ID=easy_missing_values uvicorn envs.data_cleaning_env.server.app:app --port 8000 --reload
```

### 3. Run the baselines

```bash
python baseline.py --base-url http://localhost:8000 --task easy_missing_values
```

### 4. Use the typed client in your code

```python
from envs.data_cleaning_env.client import DataCleaningEnv
from envs.data_cleaning_env.models import DataCleaningAction

env = DataCleaningEnv(base_url="http://localhost:8000")

result = env.reset()
print(f"Initial quality: {result.observation.overall_quality:.2f}")
print(f"Missing values:  {result.observation.total_missing}")

# Apply a cleaning operation
result = env.step(DataCleaningAction(
    operation="fill_missing_mean",
    column="Age"
))
print(f"Reward: {result.reward:+.3f}")
print(f"Quality now: {result.observation.overall_quality:.2f}")

# Submit when done
result = env.step(DataCleaningAction(operation="submit"))
print(f"Final grader score: {result.observation.overall_quality:.2f}")
```

---

## 🐳 Docker

```bash
# Build
docker build -t data-cleaning-env .

# Run (easy task)
docker run -e TASK_ID=easy_missing_values -p 7860:7860 data-cleaning-env

# Run (hard task)
docker run -e TASK_ID=hard_full_pipeline -p 7860:7860 data-cleaning-env
```

---

## 🤗 Hugging Face Spaces Deployment

1. Create a new Space on HF (Docker type)
2. Push this repo to the Space
3. The server starts automatically on port 7860
4. Your client connects to `https://<username>-<space-name>.hf.space`

```python
env = DataCleaningEnv(base_url="https://your-username-data-cleaning-env.hf.space")
```

---

## 📁 Project Structure

```
data_cleaning_env/
├── src/
│   ├── core/
│   │   ├── env_server.py          ← Base classes (Environment, Action, Observation)
│   │   └── http_env_client.py     ← Base HTTP client
│   └── envs/
│       └── data_cleaning_env/
│           ├── models.py          ← Typed Action, Observation, State
│           ├── tasks.py           ← 3 tasks + agent graders
│           ├── client.py          ← DataCleaningEnv (what your training imports)
│           └── server/
│               ├── environment.py ← Core simulation logic
│               └── app.py         ← FastAPI server
├── baseline.py                    ← Reproducible baseline scores
├── openenv.yaml                   ← Environment spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Baseline Scores (seed=42)

| Agent | Task | Quality | Success Rate |
|-------|------|---------|-------------|
| RandomAgent | easy | ~0.70 | ~20% |
| HeuristicAgent | easy | ~0.95 | ~100% |
| RandomAgent | medium | ~0.55 | ~0% |
| HeuristicAgent | medium | ~0.88 | ~80% |
| RandomAgent | hard | ~0.45 | ~0% |
| HeuristicAgent | hard | ~0.82 | ~60% |
