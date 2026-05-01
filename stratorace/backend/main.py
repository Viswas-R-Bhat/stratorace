# main.py
"""
StratoRace FastAPI backend
/api/simulate now uses the REAL trained PPO agent (ppo_pit_strategy_final.zip)
via stable-baselines3.
"""

import asyncio
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

import torch as th
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="StratoRace API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA        = Path(__file__).parent / "data"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# ── Compound encoding (must match training env) ───────────────────────────────
COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTER": 3}

# ── PPO model loader ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_ppo_model():
    from stable_baselines3 import PPO
    model_path = CHECKPOINTS / "ppo_pit_strategy_final.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO model not found at {model_path}. "
            "Make sure ppo_pit_strategy_final.zip is in stratorace/backend/checkpoints/"
        )
    model = PPO.load(str(model_path), device="cpu")
    return model


def build_observation(req) -> np.ndarray:
    laps_remaining = req.totalLaps - req.lap
    obs = np.array([
        req.tyreAge / 45.0,
        req.lap / max(req.totalLaps, 1),
        min(req.gapAhead / 30.0, 1.0),
        laps_remaining / max(req.totalLaps, 1),
        0.0,
        min(req.gapBehind / 35.0, 1.0),
        COMPOUND_IDX.get(req.compound, 1) / 3.0,
        (req.trackTemp - 20.0) / 280.0,
        float(req.rainfall),
    ], dtype=np.float32)
    return obs


# ── Data loaders ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_shap():
    df = pd.read_csv(DATA / "shap_values.csv")
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    mean_abs = df[shap_cols].abs().mean().sort_values(ascending=False)
    features = []
    for col, val in mean_abs.items():
        name = col.replace("shap_", "").replace("_", " ")
        features.append({"name": name, "col": col, "value": round(float(val), 4)})
    top8 = [f["col"] for f in features[:8]]
    sample = df[top8].sample(min(300, len(df)), random_state=42)
    beeswarm = []
    for feat_idx, col in enumerate(top8):
        for _, v in enumerate(sample[col]):
            beeswarm.append({"feat": feat_idx, "shap": round(float(v), 4)})
    return {"features": features, "beeswarm": beeswarm}


@lru_cache(maxsize=1)
def load_evaluation():
    return pd.read_csv(DATA / "agent_full_evaluation.csv")


@lru_cache(maxsize=1)
def load_training():
    npz = np.load(DATA / "evaluations.npz")
    timesteps  = npz["timesteps"].tolist()
    results    = npz["results"]
    ep_lengths = npz["ep_lengths"]
    return {
        "timesteps":   timesteps,
        "mean_reward": [round(v, 2) for v in results.mean(axis=1).tolist()],
        "min_reward":  [round(v, 2) for v in results.min(axis=1).tolist()],
        "max_reward":  [round(v, 2) for v in results.max(axis=1).tolist()],
        "mean_ep_len": [round(v, 1) for v in ep_lengths.mean(axis=1).tolist()],
    }


@lru_cache(maxsize=1)
def load_tyre_model():
    return pd.read_csv(DATA / "tyre_model_per_compound.csv").to_dict(orient="records")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: Optional[str] = None

class SimRequest(BaseModel):
    compound:   str
    tyreAge:    int
    lap:        int
    totalLaps:  int
    position:   int
    gapAhead:   float
    gapBehind:  float
    trackTemp:  float
    rainfall:   bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "StratoRace API v2 — real PPO"}


# ── /api/chat ─────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    contents = []
    for m in req.messages:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})

    system_text = req.system or (
        "You are the StratoRace AI assistant — an F1 pit strategy optimisation system "
        "built on a PPO reinforcement learning agent trained on 2022–2024 Formula 1 data. "
        "Keep answers concise (2-4 sentences). Only answer about StratoRace, F1 strategy, "
        "tyre behaviour, or the dashboard data."
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7}
    }

    MAX_RETRIES = 4
    BASE_DELAY  = 2.0

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(MAX_RETRIES):
            r = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers={"content-type": "application/json"},
            )

            if r.status_code == 200:
                break

            if r.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue
                else:
                    raise HTTPException(
                        status_code=429,
                        detail="Gemini rate limit hit after 4 retries. Please wait a moment and try again."
                    )
            else:
                raise HTTPException(status_code=r.status_code, detail=r.text)

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = "No response from model."
    return {"text": text}


# ── /api/simulate — REAL PPO INFERENCE ───────────────────────────────────────
@app.post("/api/simulate")
def simulate(req: SimRequest):
    try:
        model = load_ppo_model()
        obs   = build_observation(req)

        action, _ = model.predict(obs, deterministic=True)
        pit = bool(int(action) == 1)

        obs_tensor = th.tensor(obs[None, :], dtype=th.float32)
        with th.no_grad():
            dist  = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().numpy()

        pit_prob  = float(probs[1]) if len(probs) > 1 else (0.8 if pit else 0.2)
        stay_prob = float(probs[0]) if len(probs) > 1 else (1 - pit_prob)

        confidence    = int(round((pit_prob if pit else stay_prob) * 100))
        altConfidence = 100 - confidence

        tyre_r2   = {"HARD": 0.820534, "MEDIUM": 0.726092, "SOFT": 0.535472, "INTER": 0.65}
        deg_rate  = {"SOFT": 0.18, "MEDIUM": 0.09, "HARD": 0.05, "INTER": 0.12}
        cliff_lap = {"SOFT": 14, "MEDIUM": 22, "HARD": 30, "INTER": 18}

        comp      = req.compound
        laps_left = req.totalLaps - req.lap
        deg_score = req.tyreAge * deg_rate.get(comp, 0.09)
        cliff     = cliff_lap.get(comp, 22)
        cliff_bonus = float(np.power(max(0, req.tyreAge - cliff), 1.4) * 0.15) if req.tyreAge > cliff else 0.0
        urgency   = deg_score + cliff_bonus + (1.8 if req.gapAhead > 8 else 0) + (2.5 if laps_left < 12 else 0) + (4.0 if req.rainfall and comp != "INTER" else 0)

        next_comp_map = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "MEDIUM", "INTER": "MEDIUM"}
        rec_compound  = next_comp_map[comp] if pit else comp
        optimal_out   = {"SOFT": 12, "MEDIUM": 20, "HARD": 28, "INTER": 15}[comp]

        return {
            "pit":           pit,
            "confidence":    confidence,
            "altConfidence": altConfidence,
            "pitProb":       round(pit_prob, 4),
            "stayProb":      round(stay_prob, 4),
            "degScore":      round(deg_score, 3),
            "cliffBonus":    round(cliff_bonus, 3),
            "urgency":       round(urgency, 3),
            "lapsLeft":      laps_left,
            "recCompound":   rec_compound,
            "windowStart":   max(1, optimal_out - 4),
            "windowEnd":     min(req.totalLaps, optimal_out + 6),
            "tyreModelR2":   round(tyre_r2.get(comp, 0.7), 3),
            "modelSource":   "ppo_agent",
        }

    except FileNotFoundError as e:
        return _heuristic_simulate(req, warning=str(e))
    except Exception as e:
        return _heuristic_simulate(req, warning=f"Model error: {e}")


def _heuristic_simulate(req: SimRequest, warning: str = "") -> dict:
    tyre_r2   = {"HARD": 0.820534, "MEDIUM": 0.726092, "SOFT": 0.535472, "INTER": 0.65}
    deg_rate  = {"SOFT": 0.18, "MEDIUM": 0.09, "HARD": 0.05, "INTER": 0.12}
    cliff_lap = {"SOFT": 14, "MEDIUM": 22, "HARD": 30, "INTER": 18}

    comp      = req.compound
    r2        = tyre_r2.get(comp, 0.7)
    laps_left = req.totalLaps - req.lap
    deg_score = req.tyreAge * deg_rate[comp]
    cliff     = cliff_lap[comp]
    cliff_bonus = float(np.power(max(0, req.tyreAge - cliff), 1.4) * 0.15) if req.tyreAge > cliff else 0.0
    gap_signal  = 1.8 if req.gapAhead > 8 else 0.0
    late_signal = 2.5 if laps_left < 12 else 0.0
    rain_signal = 4.0 if req.rainfall and comp != "INTER" else 0.0
    urgency     = deg_score + cliff_bonus + gap_signal + late_signal + rain_signal
    PIT_THRESHOLD = 3.5
    pit         = bool(urgency > PIT_THRESHOLD)
    raw_gap     = abs(urgency - PIT_THRESHOLD)
    base_conf   = 55 + raw_gap * 14
    confidence  = int(min(96, max(51, round(base_conf * (0.7 + 0.3 * r2)))))
    next_comp_map = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "MEDIUM", "INTER": "MEDIUM"}
    optimal_out   = {"SOFT": 12, "MEDIUM": 20, "HARD": 28, "INTER": 15}[comp]

    result = {
        "pit":           pit,
        "confidence":    confidence,
        "altConfidence": 100 - confidence,
        "degScore":      round(deg_score, 3),
        "cliffBonus":    round(cliff_bonus, 3),
        "urgency":       round(urgency, 3),
        "lapsLeft":      laps_left,
        "recCompound":   next_comp_map[comp] if pit else comp,
        "windowStart":   max(1, optimal_out - 4),
        "windowEnd":     min(req.totalLaps, optimal_out + 6),
        "tyreModelR2":   round(r2, 3),
        "modelSource":   "heuristic_fallback",
    }
    if warning:
        result["warning"] = warning
    return result


# ── /api/shap ─────────────────────────────────────────────────────────────────
@app.get("/api/shap")
def shap():
    return load_shap()

# ── /api/training ─────────────────────────────────────────────────────────────
@app.get("/api/training")
def training():
    return load_training()

# ── /api/evaluation ───────────────────────────────────────────────────────────
@app.get("/api/evaluation")
def evaluation(year: Optional[int] = None, gp: Optional[str] = None, driver: Optional[str] = None):
    df = load_evaluation()
    if year:   df = df[df["year"] == year]
    if gp:     df = df[df["gp"] == gp]
    if driver: df = df[df["driver"] == driver]
    gp_agg = (
        df.groupby("gp")
        .agg(mean_reward=("total_reward","mean"), mean_delta=("mean_delta","mean"),
             mean_pit_error=("pit_timing_error","mean"), count=("driver","count"))
        .reset_index().sort_values("mean_reward", ascending=False)
    )
    total   = len(df)
    correct = int((df["total_reward"] > 0).sum())
    return {
        "stats": {
            "total_races":        total,
            "correct_calls":      correct,
            "accuracy_pct":       round(correct / total * 100, 1) if total else 0,
            "avg_pos_gain":       round(float(df["mean_delta"].mean()), 2),
            "avg_pit_error_laps": round(float(df["pit_timing_error"].abs().mean()), 1),
        },
        "by_gp": gp_agg.to_dict(orient="records"),
        "rows":  df.head(50).fillna("").to_dict(orient="records"),
    }

# ── /api/tyre-model ───────────────────────────────────────────────────────────
@app.get("/api/tyre-model")
def tyre_model():
    return {"compounds": load_tyre_model()}

# ── /api/races ────────────────────────────────────────────────────────────────
@app.get("/api/races")
def races():
    df = load_evaluation()
    return {
        "years":   sorted(df["year"].unique().tolist()),
        "gps":     sorted(df["gp"].unique().tolist()),
        "drivers": sorted(df["driver"].unique().tolist()),
    }
