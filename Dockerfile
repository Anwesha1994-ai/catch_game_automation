# envs/data_cleaning_env/server/Dockerfile
# -----------------------------------------
# Builds the Data Cleaning OpenEnv server.
#
# Build:   docker build -t data-cleaning-env .
# Run:     docker run -e TASK_ID=easy_missing_values -p 8000:8000 data-cleaning-env
# HF Space: push this Dockerfile + src/ to your HF Space repo

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ──────────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Environment variables (override at runtime) ───────────────────────────────
ENV TASK_ID=easy_missing_values
ENV PYTHONPATH=/app/src

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Start the server ──────────────────────────────────────────────────────────
# Port 7860 is required by Hugging Face Spaces
CMD ["uvicorn", "envs.data_cleaning_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]
