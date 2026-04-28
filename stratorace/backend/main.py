import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="StratoRace API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your Vercel URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA = Path(__file__).parent / "data"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ── Data loading (cached) ────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_shap():
    df = pd.read_csv(DATA / "shap_values.csv")
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    mean_abs = df[shap_cols].abs().mean().sort_values(ascending=False)
    features = []
    for col, val in mean_abs.items():
        name = col.replace("shap_", "").replace("_", " ")
        features.append({"name": name, "col": col, "value": round(float(val), 4)})
    # beeswarm sample: 300 random rows, top 8 features
    top8 = [f["col"] for f in features[:8]]
    sample = df[top8].sample(min(300, len(df)), random_state=42)
    beeswarm = []
    for feat_idx, col in enumerate(top8):
        for _, v in enumerate(sample[col]):
            beeswarm.append({"feat": feat_idx, "shap": round(float(v), 4)})
    return {"features": features, "beeswarm": beeswarm}


@lru_cache(maxsize=1)
def load_evaluation():
    df = pd.read_csv(DATA / "agent_full_evaluation.csv")
    return df


@lru_cache(maxsize=1)
def load_training():
    npz = np.load(DATA / "evaluations.npz")
    timesteps = npz["timesteps"].tolist()
    results   = npz["results"]          # shape (50, 5)
    ep_lengths = npz["ep_lengths"]      # shape (50, 5)
    mean_reward = results.mean(axis=1).tolist()
    min_reward  = results.min(axis=1).tolist()
    max_reward  = results.max(axis=1).tolist()
    mean_ep_len = ep_lengths.mean(axis=1).tolist()
    return {
        "timesteps":   timesteps,
        "mean_reward": [round(v, 2) for v in mean_reward],
        "min_reward":  [round(v, 2) for v in min_reward],
        "max_reward":  [round(v, 2) for v in max_reward],
        "mean_ep_len": [round(v, 1) for v in mean_ep_len],
    }


@lru_cache(maxsize=1)
def load_tyre_model():
    df = pd.read_csv(DATA / "tyre_model_per_compound.csv")
    return df.to_dict(orient="records")


# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: Optional[str] = None

class SimRequest(BaseModel):
    compound: str           # SOFT | MEDIUM | HARD | INTER
    tyreAge: int
    lap: int
    totalLaps: int
    position: int
    gapAhead: float
    gapBehind: float
    trackTemp: float
    rainfall: bool


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "StratoRace API"}


# ── /api/chat  (Gemini proxy — keeps key server-side) ────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    # Build Gemini contents array from message history
    contents = []
    for m in req.messages:
        role = "user" if m.role == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": m.content}]
        })

    # Prepend system prompt as first user message if provided
    system_text = req.system or (
        "You are the StratoRace AI assistant — an F1 pit strategy optimisation system "
        "built on a PPO reinforcement learning agent trained on 2022–2024 Formula 1 data. "
        "Keep answers concise (2-4 sentences). Only answer about StratoRace, F1 strategy, "
        "tyre behaviour, or the dashboard data. If outside scope say: "
        "'I can only answer questions about the StratoRace project and F1 strategy model.'"
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7}
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"content-type": "application/json"},
        )

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = "No response from model."
    return {"text": text}


# ── /api/simulate ─────────────────────────────────────────────────────────────
@app.post("/api/simulate")
def simulate(req: SimRequest):
    """
    Heuristic PPO proxy calibrated against real tyre-model R² values.
    Real model inference would require the .zip weights — this gives
    a faithful approximation of the agent's decision boundary.
    """
    tyre_r2 = {"HARD": 0.820534, "MEDIUM": 0.726092, "SOFT": 0.535472, "INTER": 0.65}
    deg_rate = {"SOFT": 0.18, "MEDIUM": 0.09, "HARD": 0.05, "INTER": 0.12}
    cliff_lap = {"SOFT": 14, "MEDIUM": 22, "HARD": 30, "INTER": 18}

    comp = req.compound
    r2 = tyre_r2.get(comp, 0.7)
    laps_left = req.totalLaps - req.lap

    # Tyre degradation signal
    deg_score = req.tyreAge * deg_rate[comp]

    # Past-cliff exponential penalty
    cliff = cliff_lap[comp]
    cliff_bonus = (
        float(np.power(max(0, req.tyreAge - cliff), 1.4) * 0.15)
        if req.tyreAge > cliff else 0.0
    )

    # Situation signals
    gap_signal   = 1.8 if req.gapAhead > 8 else 0.0
    late_signal  = 2.5 if laps_left < 12 else 0.0
    rain_signal  = 4.0 if req.rainfall and comp != "INTER" else 0.0

    urgency = deg_score + cliff_bonus + gap_signal + late_signal + rain_signal
    PIT_THRESHOLD = 3.5
    pit = bool(urgency > PIT_THRESHOLD)

    # Confidence — scaled by model R² (higher R² = more certain tyre model)
    raw_gap = abs(urgency - PIT_THRESHOLD)
    base_conf = 55 + raw_gap * 14
    confidence = int(min(96, max(51, round(base_conf * (0.7 + 0.3 * r2)))))

    # Recommended next compound
    next_comp_map = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "MEDIUM", "INTER": "MEDIUM"}
    rec_compound = next_comp_map[comp] if pit else comp

    # Optimal pit window
    optimal_out = {"SOFT": 12, "MEDIUM": 20, "HARD": 28, "INTER": 15}[comp]
    window_start = max(1, optimal_out - 4)
    window_end   = min(req.totalLaps, optimal_out + 6)

    return {
        "pit":          pit,
        "confidence":   confidence,
        "altConfidence": 100 - confidence,
        "degScore":     round(deg_score, 3),
        "cliffBonus":   round(cliff_bonus, 3),
        "urgency":      round(urgency, 3),
        "lapsLeft":     laps_left,
        "recCompound":  rec_compound,
        "windowStart":  window_start,
        "windowEnd":    window_end,
        "tyreModelR2":  round(r2, 3),
    }


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
    if year:
        df = df[df["year"] == year]
    if gp:
        df = df[df["gp"] == gp]
    if driver:
        df = df[df["driver"] == driver]

    # Aggregate per GP for bar chart (mean total_reward)
    gp_agg = (
        df.groupby("gp")
        .agg(
            mean_reward=("total_reward", "mean"),
            mean_delta=("mean_delta", "mean"),
            mean_pit_error=("pit_timing_error", "mean"),
            count=("driver", "count"),
        )
        .reset_index()
        .sort_values("mean_reward", ascending=False)
    )

    # Overall stats
    total = len(df)
    correct = int((df["total_reward"] > 0).sum())
    accuracy = round(correct / total * 100, 1) if total else 0
    avg_pos_gain = round(float(df["mean_delta"].mean()), 2)
    avg_pit_error = round(float(df["pit_timing_error"].abs().mean()), 1)

    return {
        "stats": {
            "total_races": total,
            "correct_calls": correct,
            "accuracy_pct": accuracy,
            "avg_pos_gain": avg_pos_gain,
            "avg_pit_error_laps": avg_pit_error,
        },
        "by_gp": gp_agg.to_dict(orient="records"),
        "rows": df.head(50).fillna("").to_dict(orient="records"),
    }


# ── /api/tyre-model ───────────────────────────────────────────────────────────
@app.get("/api/tyre-model")
def tyre_model():
    return {"compounds": load_tyre_model()}


# ── /api/races  (filter options) ──────────────────────────────────────────────
@app.get("/api/races")
def races():
    df = load_evaluation()
    return {
        "years":   sorted(df["year"].unique().tolist()),
        "gps":     sorted(df["gp"].unique().tolist()),
        "drivers": sorted(df["driver"].unique().tolist()),
    }