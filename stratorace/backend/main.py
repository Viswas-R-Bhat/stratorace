"""
StratoRace — Railway Backend (FastAPI)
Endpoints:
  GET  /health        — liveness check
  POST /api/chat      — proxies messages to Anthropic (API key stays server-side)
  POST /api/predict   — runs PPO model inference, returns action + probabilities
"""

import os
from pathlib import Path

import numpy as np
import torch
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="StratoRace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tightened: set to your Vercel domain in prod
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Load PPO model once at startup ────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/ppo_pit_strategy_final.zip")

print(f"[StratoRace] Loading PPO model from: {MODEL_PATH}")
try:
    ppo_model = PPO.load(MODEL_PATH)
    print(f"[StratoRace] Model ready — obs: {ppo_model.observation_space}  actions: {ppo_model.action_space}")
except Exception as e:
    print(f"[StratoRace] WARNING: could not load model — {e}")
    ppo_model = None

# ── Action metadata ───────────────────────────────────────────────────────────
# Verified by probing the trained model on known race scenarios:
#   0 → EMERGENCY PIT  (rain on wrong compound / cliff degradation)
#   1 → PIT NOW        (standard degradation-driven stop)
#   2 → MONITOR        (marginal — near window but not critical)
#   3 → STAY OUT       (tyres healthy, remain on track)
ACTION_LABELS  = {0: "EMERGENCY PIT", 1: "PIT NOW", 2: "MONITOR", 3: "STAY OUT"}
ACTION_COLOURS = {0: "red",           1: "lime",    2: "yellow",  3: "orange"}

COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTER": 3}
NEXT_COMPOUND = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "MEDIUM", "INTER": "MEDIUM"}
OPTIMAL_LAPS  = {"SOFT": 14,       "MEDIUM": 22,    "HARD": 32,       "INTER": 18}
DEGRADE_RATE  = {"SOFT": 0.10,     "MEDIUM": 0.05,  "HARD": 0.025,    "INTER": 0.07}

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    compound:   str   = "MEDIUM"
    tyreAge:    float = 15
    lap:        float = 30
    totalLaps:  float = 52
    position:   float = 10
    gapAhead:   float = 5
    trackTemp:  float = 38
    rainfall:   bool  = False
    speedSt:    float = 170     # optional — defaults to mid-range

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system:   str = ""

# ── Observation builder ───────────────────────────────────────────────────────
def build_obs(req: PredictRequest) -> np.ndarray:
    """
    Converts raw inputs to the 9-feature vector the PPO model expects.
    Observation space: Box(0.0, 2.0, shape=(9,), dtype=float32)

    Feature order (must match training env in src/rl_pit_strategy.py):
      0  tyreAge       / 45  * 2
      1  lapTimeDelta  / 5   * 2   (estimated from tyre age × compound rate)
      2  gapAhead      / 60  * 2
      3  lapsRemaining / 78  * 2
      4  position      / 20  * 2
      5  compound_idx  / 3   * 2   (SOFT=0, MEDIUM=1, HARD=2, INTER=3)
      6  trackTemp     / 60  * 2
      7  rainfall      * 2         (0 or 1 → 0.0 or 2.0)
      8  speedST       / 340 * 2
    """
    rate            = DEGRADE_RATE.get(req.compound, 0.05)
    lap_time_delta  = min(req.tyreAge * rate, 5.0)
    compound_idx    = COMPOUND_IDX.get(req.compound, 1)
    laps_remaining  = req.totalLaps - req.lap

    raw = np.array([
        req.tyreAge      / 45.0,
        lap_time_delta   / 5.0,
        req.gapAhead     / 60.0,
        laps_remaining   / 78.0,
        req.position     / 20.0,
        compound_idx     / 3.0,
        req.trackTemp    / 60.0,
        float(req.rainfall),
        req.speedSt      / 340.0,
    ], dtype=np.float32) * 2.0

    return np.clip(raw, 0.0, 2.0)


def get_probs(obs: np.ndarray) -> list[float]:
    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        dist  = ppo_model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.numpy()[0]
    return [round(float(p), 4) for p in probs]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ppo_model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    if ppo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH.")

    obs = build_obs(req)

    action_arr, _ = ppo_model.predict(obs[np.newaxis], deterministic=True)
    action         = int(action_arr[0])
    probs          = get_probs(obs)
    confidence     = round(probs[action] * 100, 1)

    # Pit window (based on compound optimal lap)
    opt         = OPTIMAL_LAPS.get(req.compound, 22)
    win_start   = max(1, opt - 4)
    win_end     = min(int(req.totalLaps), opt + 8)

    rec_compound = NEXT_COMPOUND.get(req.compound, "MEDIUM") if action in (0, 1) else req.compound

    return {
        "action":      action,
        "label":       ACTION_LABELS[action],
        "colour":      ACTION_COLOURS[action],
        "confidence":  confidence,
        "probs": {
            "EMERGENCY_PIT": round(probs[0] * 100, 1),
            "PIT_NOW":       round(probs[1] * 100, 1),
            "MONITOR":       round(probs[2] * 100, 1),
            "STAY_OUT":      round(probs[3] * 100, 1),
        },
        "recCompound": rec_compound,
        "pitWindow":   {"start": win_start, "end": win_end},
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Proxy to Anthropic — keeps ANTHROPIC_API_KEY server-side.
    Set ANTHROPIC_API_KEY in Railway environment variables.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server.")

    payload = {
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages":   [m.model_dump() for m in req.messages],
    }
    if req.system:
        payload["system"] = req.system

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json=payload,
        )

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    text = next((b["text"] for b in data.get("content", []) if b["type"] == "text"), "")
    return {"text": text}
