"""
envs/catch_env/tasks.py
--------------------------------
Three tasks of increasing difficulty.  Each task defines:
  - A messy dataset (generated procedurally so it's reproducible)
  - A description of what the agent must fix
  - A grader function: (final_observation) → float score 0.0–1.0
  - A target quality threshold

Task structure:
  EASY   → 1 problem type  (missing values only)
  MEDIUM → 2 problem types (missing + type errors)
  HARD   → 3 problem types (missing + type errors + outliers + duplicates)
"""

import io
import random
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task definition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str              # "easy" | "medium" | "hard"
    description: str
    target_quality: float        # agent must reach this to succeed
    max_steps: int
    generate_dataset: Callable[[], pd.DataFrame]   # returns messy df
    grader: Callable[[dict], float]                # obs dict → 0.0–1.0


# ---------------------------------------------------------------------------
# TASK 1 — EASY
# Problem: Missing values only
# Dataset: Employee records with NaN salaries and ages
# ---------------------------------------------------------------------------

def _generate_easy_dataset() -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    n = 60
    data = {
        "EmployeeID": range(1001, 1001 + n),
        "Name": [f"Employee_{i}" for i in range(n)],
        "Department": np.random.choice(["HR", "Engineering", "Sales", "Finance"], n),
        "Age": np.where(
            np.random.random(n) < 0.20,   # 20% missing
            np.nan,
            np.random.randint(22, 65, n).astype(float)
        ),
        "Salary": np.where(
            np.random.random(n) < 0.15,   # 15% missing
            np.nan,
            np.random.randint(30000, 120000, n).astype(float)
        ),
    }
    return pd.DataFrame(data)


def _grade_easy(obs: dict) -> float:
    """
    Score based purely on completeness.
    Full marks if no missing values remain.
    """
    completeness = obs.get("completeness_score", 0.0)
    overall = obs.get("overall_quality", 0.0)
    # Weighted: completeness matters most for easy task
    return round(0.7 * completeness + 0.3 * overall, 4)


TASK_EASY = Task(
    task_id="easy_missing_values",
    difficulty="easy",
    description=(
        "Clean an employee dataset with missing values. "
        "Fill in missing Age values with the column mean and "
        "missing Salary values with the column mean. "
        "Achieve a completeness score ≥ 0.95."
    ),
    target_quality=0.90,
    max_steps=15,
    generate_dataset=_generate_easy_dataset,
    grader=_grade_easy,
)


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# Problems: Missing values + type errors + inconsistent strings
# Dataset: Customer records with multiple issues
# ---------------------------------------------------------------------------

def _generate_medium_dataset() -> pd.DataFrame:
    random.seed(7)
    np.random.seed(7)
    n = 80
    genders = ["Male", "Female", "male", "FEMALE", "M", "F", "  Male  ", "female "]
    data = {
        "CustomerID": range(2001, 2001 + n),
        "Gender": np.random.choice(genders, n),   # inconsistent strings
        "Age": np.where(
            np.random.random(n) < 0.18,
            np.nan,
            np.random.randint(18, 70, n).astype(float)
        ),
        # Salary stored as string with commas — wrong dtype
        "Income": [
            f"{v:,}" if random.random() > 0.1 else str(v)
            for v in np.random.randint(20000, 150000, n)
        ],
        "City": [
            f"  {c}  " if random.random() < 0.3 else c
            for c in np.random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai"], n)
        ],
        "Churn": np.random.choice([0, 1], n),
    }
    return pd.DataFrame(data)


def _grade_medium(obs: dict) -> float:
    """
    Score across completeness + consistency (dtype correctness).
    Both must be high for a good score.
    """
    completeness = obs.get("completeness_score", 0.0)
    consistency  = obs.get("consistency_score", 0.0)
    uniqueness   = obs.get("uniqueness_score", 1.0)
    # Equal weight for medium task
    return round((completeness + consistency + uniqueness) / 3, 4)


TASK_MEDIUM = Task(
    task_id="medium_type_and_string_errors",
    difficulty="medium",
    description=(
        "Clean a customer dataset with three types of issues:\n"
        "1. Missing Age values (fill with mean).\n"
        "2. Income column is a string with commas — cast it to int64.\n"
        "3. Gender values are inconsistent (Male/male/M/FEMALE) — "
        "normalize to title case.\n"
        "4. City names have leading/trailing whitespace — strip them.\n"
        "Achieve overall quality ≥ 0.85."
    ),
    target_quality=0.85,
    max_steps=20,
    generate_dataset=_generate_medium_dataset,
    grader=_grade_medium,
)


# ---------------------------------------------------------------------------
# TASK 3 — HARD
# Problems: ALL of the above + outliers + duplicate rows
# Dataset: Sales transactions — the messiest of all
# ---------------------------------------------------------------------------

def _generate_hard_dataset() -> pd.DataFrame:
    random.seed(99)
    np.random.seed(99)
    n = 120

    quantities = np.random.randint(1, 50, n).astype(float)
    # Inject 8% outliers
    outlier_idx = np.random.choice(n, int(0.08 * n), replace=False)
    quantities[outlier_idx] = np.random.randint(500, 1000, len(outlier_idx))

    prices = np.random.uniform(5.0, 500.0, n)
    prices[np.random.choice(n, int(0.08 * n), replace=False)] = -999.0  # bad values

    products = np.random.choice(
        ["Laptop", "laptop", "LAPTOP", "Phone", "phone", "Tablet", " Tablet"], n
    )

    data = {
        "TransactionID": list(range(3001, 3001 + n)),
        "Product": products,                           # inconsistent case
        "Quantity": np.where(np.random.random(n) < 0.12, np.nan, quantities),
        "UnitPrice": prices,                           # negative outliers
        "SalesRep": [f"  Rep_{r}  " if random.random() < 0.4
                     else f"Rep_{r}"
                     for r in np.random.randint(1, 10, n)],  # whitespace
        "Region": [
            None if np.random.random() < 0.10
            else r
            for r in np.random.choice(["North", "South", "East", "West"], n)
        ],
    }
    df = pd.DataFrame(data)

    # Add ~10% duplicate rows
    dup_count = int(0.10 * n)
    duplicates = df.sample(dup_count, random_state=1)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def _grade_hard(obs: dict) -> float:
    """
    Strict grader: all three quality dimensions must be high.
    Penalty if duplicate_rows or total_missing remain high.
    """
    completeness = obs.get("completeness_score", 0.0)
    consistency  = obs.get("consistency_score", 0.0)
    uniqueness   = obs.get("uniqueness_score", 0.0)

    base_score = 0.4 * completeness + 0.3 * consistency + 0.3 * uniqueness

    # Extra penalty if many duplicates remain
    dup_ratio = obs.get("duplicate_rows", 0) / max(obs.get("num_rows", 1), 1)
    penalty = min(0.2, dup_ratio)

    return round(max(0.0, base_score - penalty), 4)


TASK_HARD = Task(
    task_id="hard_full_pipeline",
    difficulty="hard",
    description=(
        "Clean a sales transaction dataset with ALL types of issues:\n"
        "1. Missing Quantity and Region values.\n"
        "2. Negative UnitPrice values (outliers/errors) — clip to [0, mean+2std].\n"
        "3. Inconsistent Product names (Laptop/laptop/LAPTOP) — normalize.\n"
        "4. SalesRep column has whitespace — strip it.\n"
        "5. ~10% duplicate rows — drop them.\n"
        "Achieve overall quality ≥ 0.90."
    ),
    target_quality=0.90,
    max_steps=30,
    generate_dataset=_generate_hard_dataset,
    grader=_grade_hard,
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

ALL_TASKS: Dict[str, Task] = {
    TASK_EASY.task_id:   TASK_EASY,
    TASK_MEDIUM.task_id: TASK_MEDIUM,
    TASK_HARD.task_id:   TASK_HARD,
}
