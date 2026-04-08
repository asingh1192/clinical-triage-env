"""
FastAPI server for ClinicalTriageEnv.
Exposes: POST /reset, POST /step, GET /state, GET /health
Runs on port 7860 (Hugging Face Spaces default).
"""
from __future__ import annotations

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from env.clinical_env import ClinicalTriageEnv
from env.models import TriageAction

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ClinicalTriageEnv API",
    description=(
        "OpenEnv-compatible HTTP API for the ClinicalTriage reinforcement "
        "learning environment. Simulates hospital emergency department triage "
        "and discharge planning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Single global environment instance (stateful resets per episode)
_env = ClinicalTriageEnv(seed=42)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None, description="Optional seed for reproducibility")
    task: Optional[str] = Field(default=None, description="Optional task ID")


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health():
    """Liveness probe — always returns 200."""
    return {"status": "ok"}


@app.post("/reset", tags=["env"])
async def reset(task: Optional[str] = None, seed: Optional[int] = None, request: Optional[ResetRequest] = None):
    """
    Start a new episode.
    Returns initial PatientObservation JSON.
    """
    final_seed = seed
    final_task = task
    if request:
        if getattr(request, "seed", None) is not None:
            final_seed = request.seed
        if getattr(request, "task", None) is not None:
            final_task = request.task

    try:
        obs = _env.reset(seed=final_seed, task=final_task)
        return JSONResponse(content=obs.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse, tags=["env"])
async def step(action: TriageAction):
    """
    Execute a TriageAction.
    Returns (observation, reward, done, info).
    """
    try:
        obs, reward, done, info = _env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", tags=["env"])
async def state():
    """Return full current environment state."""
    try:
        return JSONResponse(content=_env.state())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--workers", type=int, default=1)
    args, _ = parser.parse_known_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port, workers=args.workers)

if __name__ == "__main__":
    main()
