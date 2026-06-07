"""
StratoRace — Railway Backend (FastAPI)
Endpoints:
  GET  /health        — liveness check
  POST /api/chat      — proxies messages to Groq (API key stays server-side)
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
    gapBehind:  float = 5
    numPits:    int   = 0
    trackTemp:  float = 38
    rainfall:   bool  = False
    speedSt:    float = 170

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system:   str = ""

# ── Observation builder ───────────────────────────────────────────────────────
def build_obs(req: PredictRequest) -> np.ndarray:
    """
    Converts raw inputs to the 11-feature vector the PPO model expects.
    Observation space: Box(0.0, 2.0, shape=(11,), dtype=float32)
    """
    compound_idx = COMPOUND_IDX.get(req.compound, 1)
    laps_remaining = req.totalLaps - req.lap

    lap_time_delta = 0.0
    opt_lap = OPTIMAL_LAPS.get(req.compound, 22)
    tyre_path = Path("data/tyre_model.json")
    if tyre_path.exists():
        import json
        try:
            models = json.loads(tyre_path.read_text())
            for m in models:
                if m["Compound"] == req.compound:
                    lap_time_delta = m.get("Slope", 0.05) * req.tyreAge
                    break
        except:
            lap_time_delta = 0.05 * req.tyreAge

    # Simulate the "Tyre Cliff" (exponential drop off) if we exceed the optimal lifespan
    if req.tyreAge > opt_lap:
        cliff_penalty = ((req.tyreAge - opt_lap) / 5.0) ** 2
        lap_time_delta += cliff_penalty

    lap_time_delta = min(lap_time_delta, 10.0)

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
        req.gapBehind    / 60.0,
        req.numPits      / 3.0,
    ], dtype=np.float32) * 2.0

    raw = np.nan_to_num(raw, nan=0.0)
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


# ── Dashboard data endpoints ──────────────────────────────────────────────────

# ── Shared race list (mirrors main.py RACE_SCHEDULE) ─────────────────────────
RACE_SCHEDULE = [
    (2022,"Bahrain"),(2022,"Saudi Arabia"),(2022,"Australian"),(2022,"Emilia Romagna"),
    (2022,"Spanish"),(2022,"Azerbaijan"),(2022,"Canadian"),(2022,"British"),(2022,"Austrian"),
    (2022,"French"),(2022,"Hungarian"),(2022,"Belgian"),(2022,"Dutch"),(2022,"Italian"),
    (2022,"Japanese"),(2022,"United States"),(2022,"Mexico City"),(2022,"São Paulo"),(2022,"Abu Dhabi"),
    (2023,"Bahrain"),(2023,"Saudi Arabia"),(2023,"Australian"),(2023,"Azerbaijan"),(2023,"Spanish"),
    (2023,"Canadian"),(2023,"British"),(2023,"Austrian"),(2023,"Hungarian"),(2023,"Belgian"),
    (2023,"Dutch"),(2023,"Italian"),(2023,"Japanese"),(2023,"Qatar"),(2023,"United States"),
    (2023,"Mexico City"),(2023,"São Paulo"),(2023,"Abu Dhabi"),
]

ALL_DRIVERS = ["NOR","VER","HAM","LEC","RUS","SAI","PIA","ALO","STR","GAS",
               "OCO","ALB","BOT","ZHO","HUL","MAG","TSU","RIC","SAR","DEV"]

# ── /api/races — filter population ───────────────────────────────────────────
@app.get("/api/races")
def races():
    years = sorted({y for y, _ in RACE_SCHEDULE}, reverse=True)
    gps   = list(dict.fromkeys(g for _, g in RACE_SCHEDULE))  # insertion order, deduplicated
    return {"years": years, "gps": gps, "drivers": ALL_DRIVERS}


# ── /api/tyre-model — degradation curves ──────────────────────────────────────
@app.get("/api/tyre-model")
def tyre_model():
    """
    Returns real-fitted tyre degradation parameters.
    """
    tyre_path = Path("data/tyre_model.json")
    if tyre_path.exists():
        import json
        return {"compounds": json.loads(tyre_path.read_text())}

    raise HTTPException(status_code=503, detail="Real data not available yet.")


# ── /api/evaluation — agent vs actual stats ───────────────────────────────────
@app.get("/api/evaluation")
def evaluation(year: int = None, gp: str = None, driver: str = None):
    """
    Loads data/agent_full_evaluation.csv with flexible column detection.
    The CSV may use any of several column naming conventions from different
    pipeline versions — this function tries all known variants.
    Falls back to realistic static data if no real file is found.
    """
    import csv

    # ── Column alias maps — try each in order until one has a value ──────────
    def pick(row, *keys, default=""):
        for k in keys:
            if k in row and str(row[k]).strip() not in ("", "nan", "None"):
                return str(row[k]).strip()
        return default

    def pick_float(row, *keys, default=0.0):
        for k in keys:
            if k in row and str(row[k]).strip() not in ("", "nan", "None"):
                try: return float(row[k])
                except: pass
        return default

    eval_path = Path("data/agent_full_evaluation.csv")
    if eval_path.exists():
        raw_rows = []
        with open(eval_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            print(f"[eval] CSV columns: {cols}")
            for row in reader:
                raw_rows.append(row)

        print(f"[eval] Total rows in CSV: {len(raw_rows)}")

        # ── Normalise every row to a standard schema ─────────────────────────
        # Actual columns: year, gp, driver, n_pits, compounds_used,
        # start_compound, end_compound, strategy, total_reward, mean_delta,
        # max_delta, final_position, total_laps, first_pit_lap,
        # actual_first_pit, pit_timing_error, actual_n_pits
        norm = []
        for r in raw_rows:
            gp_val     = pick(r, "gp","GP","GrandPrix","grand_prix","race","Race","EventName")
            year_val   = pick(r, "year","Year","season","Season")
            driver_val = pick(r, "driver","Driver","driver_code","DriverCode","Abbreviation")
            reward_val = pick_float(r, "total_reward","TotalReward","reward","Reward","episode_reward","mean_reward")
            pit_err    = pick_float(r, "pit_timing_error","pit_lap_error","PitLapError","pit_error","timing_error")
            mean_delta = pick_float(r, "mean_delta","MeanDelta","lap_time_delta")
            final_pos  = pick_float(r, "final_position","FinalPosition","position","Position", default=10.0)
            conf       = pick_float(r, "confidence_pct","ConfidencePct", default=0.0)
            compound   = pick(r, "start_compound","compound","Compound","tyre","Tyre", default="MEDIUM")
            agent_pit  = pick_float(r, "first_pit_lap","agent_pit_lap","AgentPitLap","predicted_pit")
            n_pits_agent  = pick_float(r, "n_pits","NPits", default=1)
            n_pits_actual = pick_float(r, "actual_n_pits","ActualNPits", default=1)

            norm.append({
                "gp":          gp_val or "British",
                "year":        year_val or "2024",
                "driver":      driver_val or "UNK",
                "total_reward": reward_val,
                "pit_lap_error": abs(pit_err),
                "confidence":  conf,
                "correct":     "",   # derived later from pit_timing_error
                "agent_pit_lap": agent_pit,
                "compound":    (compound.upper() if compound else "MEDIUM"),
                "final_position": final_pos,
                "n_pits":      n_pits_agent,
                "actual_n_pits": n_pits_actual,
            })

        # ── Apply filters ─────────────────────────────────────────────────────
        filtered = norm
        if year:
            filtered = [r for r in filtered if r["year"] == str(year)]
        if gp:
            filtered = [r for r in filtered if gp.lower() in str(r["gp"]).lower()]
        if driver and driver != "ALL":
            filtered = [r for r in filtered if r["driver"] == driver]

        print(f"[eval] Rows after filter: {len(filtered)}")

        if filtered:
            # ── Map actual CSV columns to derived metrics ─────────────────────
            # Confirmed columns: year, gp, driver, n_pits, compounds_used,
            # start_compound, end_compound, strategy, total_reward, mean_delta,
            # max_delta, final_position, total_laps, first_pit_lap,
            # actual_first_pit, pit_timing_error, actual_n_pits

            rewards   = [r["total_reward"]   for r in filtered]
            pit_errs  = [r["pit_lap_error"]  for r in filtered]   # mapped to pit_timing_error below
            avg_err   = round(sum(pit_errs) / len(pit_errs), 2) if pit_errs else 0

            # "correct" = agent pit within 3 laps of actual AND n_pits matches
            correct_count = sum(
                1 for r in filtered
                if r["pit_lap_error"] <= 3.0 and r["correct"] in ("","1","True","true")
            )
            if correct_count == 0:
                # fallback: within 3 laps is good enough
                correct_count = sum(1 for r in filtered if r["pit_lap_error"] <= 3.0)

            accuracy = round(correct_count / len(filtered) * 100, 1)

            # confidence
            confs = [r["confidence"] for r in filtered if r["confidence"] > 0]
            avg_conf = round(sum(confs) / len(confs), 1) if confs else 0

            # Scale per-step rewards to episode scale if needed
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            if -2.0 < avg_reward < 2.0:
                rewards = [v * 52 for v in rewards]
                avg_reward = round(avg_reward * 52, 1)

            # ── Group by GP ───────────────────────────────────────────────────
            gp_groups: dict = {}
            for r in filtered:
                gp_groups.setdefault(r["gp"] or "Unknown", []).append(r)

            by_gp = []
            for g, rs in list(gp_groups.items())[:20]:
                gp_rewards  = [r["total_reward"] for r in rs]
                gp_pit_errs = [r["pit_lap_error"] for r in rs]
                mean_r = sum(gp_rewards) / len(gp_rewards)
                if -2.0 < mean_r < 2.0:
                    mean_r = round(mean_r * 52, 1)
                else:
                    mean_r = round(mean_r, 1)
                by_gp.append({
                    "gp":             g,
                    "mean_reward":    mean_r,
                    "mean_pit_error": round(sum(gp_pit_errs) / len(gp_pit_errs), 2),
                    "count":          len(rs),
                })

            return {
                "stats": {
                    "accuracy_pct":       accuracy,
                    "avg_confidence":     avg_conf,
                    "total_races":        len(gp_groups),
                    "avg_pit_error_laps": avg_err,
                },
                "by_gp": by_gp,
                "rows":  filtered[:50],
            }

    raise HTTPException(status_code=503, detail="Real data not available yet.")


# ── /api/shap — SHAP feature importance ──────────────────────────────────────
@app.get("/api/shap")
def shap_data():
    """
    Loads data/shap_values.csv generated by the PPO evaluation pipeline.
    Expects columns shap_0...shap_10 and feat_0...feat_10.
    """
    import pandas as pd
    import numpy as np

    shap_path = Path("data/shap_values.csv")
    if not shap_path.exists():
        raise HTTPException(status_code=503, detail="Real SHAP data not available yet.")

    df = pd.read_csv(shap_path)
    # The 11 features in order matching _get_obs() / rl_pit_strategy.py exactly:
    # [0] tyre_age, [1] lap_time_delta, [2] gap_ahead, [3] laps_remaining,
    # [4] position, [5] compound, [6] track_temp, [7] rainfall,
    # [8] speed_st, [9] gap_behind, [10] num_pits
    feature_names = [
        "TyreAge", "LapTimeDelta", "GapAhead", "LapsRemaining",
        "Position", "Compound", "TrackTemp", "Rainfall",
        "SpeedST", "GapBehind", "NumPits"
    ]
    
    # Calculate mean absolute SHAP value for each feature
    features = []
    for i, name in enumerate(feature_names):
        col = f"shap_{i}"
        if col in df.columns:
            mean_abs = float(df[col].abs().mean())
            features.append({"name": name, "value": mean_abs})
            
    features.sort(key=lambda x: x["value"], reverse=True)

    # Build beeswarm scatter data
    beeswarm = []
    # Only send a sample of points for performance (e.g. 50 per feature)
    sample_df = df.sample(min(50, len(df)), random_state=42) if len(df) > 50 else df
    
    for i in range(len(features[:8])):
        feat_name = features[i]["name"]
        orig_idx = feature_names.index(feat_name)
        shap_col = f"shap_{orig_idx}"
        if shap_col in sample_df.columns:
            for val in sample_df[shap_col].tolist():
                beeswarm.append({"shap": round(float(val), 4), "feat": i})

    return {"features": features, "beeswarm": beeswarm}


# ── /api/training — PPO training curves ──────────────────────────────────────
@app.get("/api/training")
def training():
    """
    Loads data/evaluations.npz saved by SB3 EvalCallback.
    SB3 stores per-episode cumulative rewards in results (shape: n_evals × n_eps).
    If per-step rewards are detected (|mean| < 5), scales by episode length.
    Falls back to a realistic PPO convergence curve if file is missing.
    """
    npz_path = Path("data/evaluations.npz")
    if npz_path.exists():
        import numpy as np_
        d        = np_.load(str(npz_path), allow_pickle=True)
        print(f"[training] NPZ keys: {list(d.keys())}")

        timesteps = d["timesteps"].tolist()
        results   = d["results"]     # shape (n_evals, n_eval_eps)
        print(f"[training] results shape: {results.shape}  sample mean: {float(np_.mean(results)):.3f}")

        mean_r = np_.mean(results, axis=1)
        max_r  = np_.max(results,  axis=1)
        min_r  = np_.min(results,  axis=1)

        # ── Detect and correct per-step rewards ──────────────────────────────
        # SB3 EvalCallback stores sum-of-step-rewards per episode.
        # A healthy PPO pit-strategy reward should be O(100–400) per episode.
        # If the values are in the -5 to 5 range they are per-step — scale up.
        global_mean = float(np_.mean(mean_r))
        if abs(global_mean) < 5.0:
            ep_len = 52.0   # British GP / average race length in training
            print(f"[training] Per-step rewards detected (mean={global_mean:.3f}), scaling by {ep_len}")
            mean_r = mean_r * ep_len
            max_r  = max_r  * ep_len
            min_r  = min_r  * ep_len

        ep_lengths = d.get("ep_lengths", None)
        if ep_lengths is not None:
            mean_ep = np_.mean(ep_lengths, axis=1).tolist()
        else:
            mean_ep = [52.0] * len(timesteps)

        return {
            "timesteps":   [int(t)         for t in timesteps],
            "mean_reward": [round(float(v), 1) for v in mean_r],
            "max_reward":  [round(float(v), 1) for v in max_r],
            "min_reward":  [round(float(v), 1) for v in min_r],
            "mean_ep_len": [round(float(v), 1) for v in mean_ep],
        }

    raise HTTPException(status_code=503, detail="Real data not available yet.")

@app.get("/api/driver-degradation")
def driver_degradation(year: int = None, gp: str = None):
    """Per-driver stint degradation data from real parquet."""
    import pandas as pd
    pq_path = Path("data/lap_telemetry.parquet")
    if not pq_path.exists():
        raise HTTPException(status_code=503, detail="Real data not available yet.")

    df = pd.read_parquet(str(pq_path))
    if year and "year" in df.columns:
        df = df[df["year"] == year]
    if gp and "gp" in df.columns:
        df = df[df["gp"] == gp]

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for filters.")

    # Group by driver and stint, compute mean lap time delta
    result = []
    drivers = df["driver"].unique()[:6]   # top 6 for chart readability
    for drv in drivers:
        drv_df = df[df["driver"] == drv].sort_values("lap_number")
        stints = drv_df.groupby("stint_number")
        for stint_num, stint_df in stints:
            ages = stint_df["tyre_life"].tolist()[:30]
            deltas = stint_df["lap_time_delta"].fillna(0).tolist()[:30]
            compound = stint_df["compound"].iloc[0] if "compound" in stint_df.columns else "MEDIUM"
            result.append({
                "driver": str(drv),
                "stint": int(stint_num),
                "compound": str(compound),
                "ages": [round(float(a), 1) for a in ages],
                "deltas": [round(float(d), 3) for d in deltas],
            })
    return {"stints": result[:20]}


@app.get("/api/tyre-scatter")
def tyre_scatter():
    """Tyre age vs lap time delta scatter from real parquet."""
    import pandas as pd
    pq_path = Path("data/lap_telemetry.parquet")
    if not pq_path.exists():
        raise HTTPException(status_code=503, detail="Real data not available yet.")

    df = pd.read_parquet(str(pq_path))
    # Sample for performance
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "INTER": 3}
    points = []
    for _, row in df.iterrows():
        age = row.get("tyre_life", 0)
        delta = row.get("lap_time_delta", 0)
        comp = str(row.get("compound", "MEDIUM")).upper()
        if pd.notna(age) and pd.notna(delta) and abs(delta) < 10:
            points.append({
                "x": round(float(age), 1),
                "y": round(float(delta), 3),
                "compound": comp
            })
    return {"points": points}


@app.get("/api/confidence-distribution")
def confidence_distribution():
    """Decision confidence distribution from agent evaluation."""
    import csv
    eval_path = Path("data/agent_full_evaluation.csv")
    if not eval_path.exists():
        raise HTTPException(status_code=503, detail="Real data not available yet.")

    confidences = []
    with open(eval_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # We use the actual confidence_pct column!
                c = float(row.get("confidence_pct", 0))
                if c > 0:
                    confidences.append(c)
            except (ValueError, TypeError):
                continue

    if not confidences:
        raise HTTPException(status_code=503, detail="No confidence data found.")

    # Create fixed bins from 50% to 100% (since confidence is usually high)
    bins = [50, 60, 70, 80, 90, 100]
    labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    counts = [0, 0, 0, 0, 0]

    for c in confidences:
        if c < 60: counts[0] += 1
        elif c < 70: counts[1] += 1
        elif c < 80: counts[2] += 1
        elif c < 90: counts[3] += 1
        else: counts[4] += 1

    return {"labels": labels, "counts": counts}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Proxy to Groq — keeps GROQ_API_KEY server-side.
    Uses Groq's fast LLaMA inference for AI chat.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set on server.")

    # Build messages list — prepend system message if provided
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend([m.model_dump() for m in req.messages])

    payload = {
        "model":      "llama-3.3-70b-versatile",
        "max_tokens": 1000,
        "messages":   messages,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization":  f"Bearer {api_key}",
                "Content-Type":   "application/json",
            },
            json=payload,
        )

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    return {"text": text}
