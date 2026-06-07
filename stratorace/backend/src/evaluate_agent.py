import os
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import shap
import torch

from rl_pit_strategy import F1PitStrategyEnv, COMPOUND_IDX

def evaluate_and_shap(model_path="../checkpoints/ppo_pit_strategy_final.zip", 
                      data_path="../data/lap_telemetry.parquet",
                      out_csv="../data/agent_full_evaluation.csv",
                      out_shap="../data/shap_values.csv"):
                      
    if not os.path.exists(data_path):
        print(f"Waiting for {data_path} to be generated...")
        return
        
    if not os.path.exists(model_path):
        print(f"Waiting for {model_path} to be generated...")
        return

    print("Loading data...")
    df = pd.read_parquet(data_path)
    race_groups = list(df.groupby(["year", "gp", "driver"]))
    
    print("Loading model...")
    model = PPO.load(model_path)
    
    # ── Phase 1: Evaluate agent against real pit laps ──
    print("Evaluating agent pit decisions vs real pit decisions...")
    
    results = []
    
    for (year, gp, driver), group in race_groups:
        laps = group.sort_values("lap_number").reset_index(drop=True)
        
        # Find the actual pit laps
        real_pit_laps = laps[laps["is_pit_in"]]["lap_number"].values
        if len(real_pit_laps) == 0:
            continue # No pit stops in this stint/race for this driver
            
        real_first_pit = real_pit_laps[0]
        
        # Simulate race through model to find its preferred first pit
        agent_first_pit = None
        agent_confidence = 0.0
        compound = laps.iloc[0].get("compound", "MEDIUM")
        
        # Build initial state
        tyre_age = 1
        num_pits = 0
        total_laps = laps.iloc[0].get("total_laps", 52)
        
        for idx, row in laps.iterrows():
            lap_num = row["lap_number"]
            
            # Construct observation (must match environment exactly)
            laps_remaining = total_laps - lap_num
            c_idx = COMPOUND_IDX.get(compound, 1)
            
            obs = np.array([
                tyre_age / 45.0,
                row.get("lap_time_delta", 0.0) / 5.0,
                row.get("gap_ahead", 5.0) / 60.0,
                laps_remaining / 78.0,
                row.get("position", 10) / 20.0,
                c_idx / 3.0,
                row.get("track_temp", 30.0) / 60.0,
                float(row.get("rainfall", False)),
                row.get("speed_st", 250.0) / 340.0,
                row.get("gap_behind", 5.0) / 60.0,
                num_pits / 3.0
            ], dtype=np.float32) * 2.0
            obs = np.nan_to_num(obs, nan=0.0)
            obs = np.clip(obs, 0.0, 2.0)
            
            action, _ = model.predict(obs, deterministic=True)
            
            # ── REAL confidence from action probabilities ──
            obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.numpy()[0]
            confidence = float(probs[int(action)] * 100.0)
            
            if action in [0, 1]: # Pit
                agent_first_pit = lap_num
                agent_confidence = confidence
                break
                
            tyre_age += 1
            agent_confidence = confidence  # Track last decision confidence
            
        if agent_first_pit is None:
            agent_first_pit = total_laps # Agent never pitted
            
        pit_timing_error = abs(agent_first_pit - real_first_pit)
        
        results.append({
            "year": year,
            "gp": gp,
            "driver": driver,
            "compound": compound,
            "actual_first_pit": real_first_pit,
            "agent_pit_lap": agent_first_pit,
            "first_pit_lap": agent_first_pit, # for frontend compat
            "pit_timing_error": pit_timing_error,
            "mean_reward": -pit_timing_error, # Simplified reward proxy
            "confidence_pct": round(agent_confidence, 2)  # REAL confidence!
        })
        
    eval_df = pd.DataFrame(results)
    eval_df.to_csv(out_csv, index=False)
    print(f"Saved evaluation results to {out_csv}")
    print(f"  Mean pit timing error: {eval_df['pit_timing_error'].mean():.2f} laps")
    print(f"  Mean confidence: {eval_df['confidence_pct'].mean():.1f}%")
    
    # ── Phase 2: Compute real SHAP values ──
    print("Computing SHAP values...")
    
    # Extract a random sample of observations from the dataset
    print("Building background dataset for SHAP...")
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    
    obs_list = []
    for _, row in sample_df.iterrows():
        laps_remaining = row.get("total_laps", 52) - row.get("lap_number", 1)
        c_idx = COMPOUND_IDX.get(row.get("compound", "MEDIUM"), 1)
        
        obs = np.array([
            row.get("tyre_life", 1) / 45.0,
            row.get("lap_time_delta", 0.0) / 5.0,
            row.get("gap_ahead", 5.0) / 60.0,
            laps_remaining / 78.0,
            row.get("position", 10) / 20.0,
            c_idx / 3.0,
            row.get("track_temp", 30.0) / 60.0,
            float(row.get("rainfall", False)),
            row.get("speed_st", 250.0) / 340.0,
            row.get("gap_behind", 5.0) / 60.0,
            row.get("num_pits_so_far", 0) / 3.0
        ], dtype=np.float32) * 2.0
        obs_list.append(np.nan_to_num(obs, nan=0.0))
        
    X_background = np.clip(np.array(obs_list), 0.0, 2.0)
    
    feature_names = [
        "Tyre Age", "Lap Time Delta", "Gap Ahead", "Laps Remaining", "Position", 
        "Compound", "Track Temp", "Rainfall", "Speed Trap", "Gap Behind", "Pits Made"
    ]
    
    # SHAP Explainer — explain the PIT NOW probability
    def predict_pit_prob(obs_array):
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
        with torch.no_grad():
            features = model.policy.features_extractor(obs_tensor)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            logits = model.policy.action_net(latent_pi).numpy()
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[:, 1]  # P(PIT NOW)
        
    print("Running SHAP explainer (this might take a few minutes)...")
    explainer = shap.KernelExplainer(predict_pit_prob, shap.sample(X_background, 100))
    shap_values = explainer.shap_values(X_background[:300], nsamples=100)
    
    # Save the SHAP values
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{i}" for i in range(11)])
    
    # Append the feature values themselves so we can build a beeswarm plot
    for i in range(11):
        shap_df[f"feat_{i}"] = X_background[:300, i]
        
    shap_df.to_csv(out_shap, index=False)
    print(f"Saved real SHAP values to {out_shap}")
    
    # Print SHAP summary
    print("\n" + "=" * 50)
    print("SHAP Feature Importance Summary:")
    print("=" * 50)
    for i, name in enumerate(feature_names):
        mean_abs = np.mean(np.abs(shap_values[:, i]))
        print(f"  {name:20s}  {mean_abs:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_and_shap()
