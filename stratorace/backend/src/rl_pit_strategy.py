import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch

# Constants matching main.py
COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTER": 3, "WET": 4, "UNKNOWN": 5}
NEXT_COMPOUND = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "MEDIUM", "INTER": "MEDIUM", "WET": "INTER", "UNKNOWN": "MEDIUM"}

# Compound-aware optimal stint lengths (laps) — real F1 data ranges
COMPOUND_OPTIMAL = {
    "SOFT":   (12, 20),   # Softs degrade fast, pit between lap 12-20
    "MEDIUM": (20, 30),   # Mediums are balanced
    "HARD":   (30, 45),   # Hards last longest
    "INTER":  (15, 30),   # Inters depend on drying conditions
    "WET":    (10, 25),
    "UNKNOWN": (18, 28),
}


class F1PitStrategyEnv(gym.Env):
    """
    Data-Driven F1 Pit Strategy Environment — v2 (Multi-Feature Aware).
    
    Replays real lap-by-lap telemetry from FastF1.
    Observation space: 11 normalized features [0.0, 2.0]
    
    Key improvements over v1:
    - Noise injection on lap_time_delta to prevent single-feature collapse
    - Multi-feature reward function that forces the agent to learn from
      compound, gaps, position, laps remaining, rainfall, track temp
    - Compound-aware pit windows
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, data_path="../data/lap_telemetry.parquet", meta_path="../data/race_meta.json"):
        super(F1PitStrategyEnv, self).__init__()
        
        # Action space: 0=EMERGENCY, 1=PIT NOW, 2=MONITOR, 3=STAY OUT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 11 normalized features [0.0, 2.0]
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(11,), dtype=np.float32)
        
        # Load real telemetry data
        print(f"Loading real telemetry from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing {data_path}. Run ingest_fastf1.py first.")
            
        self.df = pd.read_parquet(data_path)
        
        # Load race metadata for circuit pit loss
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.race_meta = json.load(f)
        else:
            self.race_meta = {}
            
        # Group by race and driver to create stints/episodes
        self.race_groups = list(self.df.groupby(["year", "gp", "driver"]))
        print(f"Loaded {len(self.race_groups)} unique driver races for training.")
        
        self.current_race_idx = -1
        self.current_laps = None
        self.current_lap_idx = 0
        
        self._init_state()

    def _init_state(self):
        self.total_laps = 52
        self.lap = 1
        self.compound = "MEDIUM"
        self.tyre_age = 0
        self.position = 10
        self.gap_ahead = 5.0
        self.gap_behind = 5.0
        self.track_temp = 30.0
        self.rainfall = False
        self.speed_st = 250.0
        self.lap_time_delta = 0.0
        self.num_pits = 0
        self.pit_loss_seconds = 22.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select a random real race
        self.current_race_idx = np.random.randint(len(self.race_groups))
        (year, gp, driver), group = self.race_groups[self.current_race_idx]
        
        self.current_laps = group.sort_values("lap_number").reset_index(drop=True)
        self.current_lap_idx = 0
        
        # Get pit loss for this circuit
        meta_key = f"{year}_{gp}"
        if meta_key in self.race_meta:
            self.pit_loss_seconds = self.race_meta[meta_key].get("pit_loss_seconds", 22.0)
        else:
            self.pit_loss_seconds = 22.0
            
        self._sync_state_from_data()
        
        return self._get_obs(), {}

    def _sync_state_from_data(self):
        if self.current_lap_idx >= len(self.current_laps):
            return
            
        lap_data = self.current_laps.iloc[self.current_lap_idx]
        
        self.total_laps = lap_data.get("total_laps", 52)
        self.lap = lap_data.get("lap_number", 1)
        self.compound = lap_data.get("compound", "MEDIUM")
        self.tyre_age = lap_data.get("tyre_life", 1)
        
        pos = lap_data.get("position", 10)
        if pd.notna(pos):
            self.position = pos
            
        gap_a = lap_data.get("gap_ahead", 5.0)
        if pd.notna(gap_a):
            self.gap_ahead = gap_a
            
        gap_b = lap_data.get("gap_behind", 5.0)
        if pd.notna(gap_b):
            self.gap_behind = gap_b
            
        temp = lap_data.get("track_temp", 30.0)
        if pd.notna(temp):
            self.track_temp = temp
            
        self.rainfall = bool(lap_data.get("rainfall", False))
        
        speed = lap_data.get("speed_st", 250.0)
        if pd.notna(speed):
            self.speed_st = speed
            
        delta = lap_data.get("lap_time_delta", 0.0)
        if pd.notna(delta):
            # ── NOISE INJECTION ──────────────────────────────────────────────
            # Add Gaussian noise to lap_time_delta during training so the model
            # CANNOT rely on this single feature as a perfect strategy proxy.
            # This forces the agent to cross-reference with tyre_age, compound,
            # gap_ahead, gap_behind, etc.
            self.lap_time_delta = delta + np.random.normal(0, 0.3)
            
        self.num_pits = lap_data.get("num_pits_so_far", 0)

    def _get_obs(self):
        """Builds observation. Must match main.py build_obs() exactly."""
        laps_remaining = self.total_laps - self.lap
        c_idx = COMPOUND_IDX.get(self.compound, 1)
        
        raw = np.array([
            self.tyre_age / 45.0,
            self.lap_time_delta / 5.0,
            self.gap_ahead / 60.0,
            laps_remaining / 78.0,
            self.position / 20.0,
            c_idx / 3.0,
            self.track_temp / 60.0,
            float(self.rainfall),
            self.speed_st / 340.0,
            self.gap_behind / 60.0,
            self.num_pits / 3.0
        ], dtype=np.float32) * 2.0
        
        # Replace NaNs with 0
        raw = np.nan_to_num(raw, nan=0.0)
        
        return np.clip(raw, 0.0, 2.0)

    def step(self, action):
        reward = 0.0
        laps_remaining = self.total_laps - self.lap
        opt_lo, opt_hi = COMPOUND_OPTIMAL.get(self.compound, (18, 28))
        
        # ──────────────────────────────────────────────────────────────────────
        # PIT STOP ACTIONS (0=EMERGENCY, 1=PIT_NOW)
        # ──────────────────────────────────────────────────────────────────────
        if action in [0, 1]:
            
            # 1. Pit timing agreement with real strategy (+15 / -8)
            actual_pit = False
            window_start = max(0, self.current_lap_idx - 2)
            window_end = min(len(self.current_laps), self.current_lap_idx + 3)
            for i in range(window_start, window_end):
                if self.current_laps.iloc[i].get("is_pit_in", False):
                    actual_pit = True
                    break
                    
            if actual_pit:
                reward += 15.0   # Agreed with real strategy
            else:
                reward -= 8.0    # Pitted when real driver didn't
                
            # 2. Compound-aware pit window bonus (+8 / -5)
            #    Rewards pitting within the optimal stint window for the current compound
            if opt_lo <= self.tyre_age <= opt_hi:
                reward += 8.0    # Pitted in optimal window for this compound
            elif self.tyre_age < opt_lo:
                reward -= 5.0    # Pitted too early for this compound
                
            # 3. Gap exploitation bonus (+5)
            #    Reward pitting when gap ahead is large enough for a safe undercut
            if self.gap_ahead > 3.0:
                reward += 5.0    # Safe pit window — won't lose position
            elif self.gap_ahead < 1.5:
                reward -= 3.0    # Dangerous — will lose position to car ahead
                
            # 4. Late-race pit penalty (-8)
            #    Pitting with very few laps left wastes time
            if laps_remaining < 8:
                reward -= 8.0    # Almost never worth pitting this late
            elif laps_remaining < 15:
                reward -= 3.0    # Risky late pit
                
            # 5. Excessive pit stops penalty (-15)
            if self.num_pits >= 2:
                reward -= 15.0   # 3+ stops is rarely optimal
            elif self.num_pits == 1:
                reward -= 2.0    # Small penalty for 2nd stop (acceptable)
                
            # 6. Weather response: pit for inters in rain (+10)
            if self.rainfall and self.compound in ["SOFT", "MEDIUM", "HARD"]:
                reward += 10.0   # Correct: switching to inters in rain
                
            # Execute the pit stop
            self.num_pits += 1
            self.tyre_age = 0
            self.compound = NEXT_COMPOUND.get(self.compound, "MEDIUM")
            self.lap_time_delta = self.pit_loss_seconds
            
        # ──────────────────────────────────────────────────────────────────────
        # STAY OUT / MONITOR (2=MONITOR, 3=STAY_OUT)
        # ──────────────────────────────────────────────────────────────────────
        else:
            # 1. Base pace reward (lower delta = faster = better)
            reward += (3.0 - abs(self.lap_time_delta))
            
            # 2. Tyre cliff penalty — severe if past optimal compound window
            if self.tyre_age > opt_hi:
                overshoot = self.tyre_age - opt_hi
                reward -= min(overshoot * 2.0, 15.0)  # Escalating penalty
            
            # 3. Lap time delta cliff penalty
            if self.lap_time_delta > 3.0:
                reward -= 10.0   # Extreme degradation — should have pitted
                
            # 4. Defensive awareness penalty
            #    If car behind is very close and tyres are worn, staying out is risky
            if self.gap_behind < 1.5 and self.tyre_age > opt_lo:
                reward -= 4.0    # Risk of being undercut by car behind
                
            # 5. Position-aware stint extension
            #    Top-5 cars benefit from extending stints (track position is king)
            if self.position <= 5 and self.tyre_age <= opt_hi:
                reward += 3.0    # Good: protecting track position with ok tyres
                
            # 6. Weather penalty: staying on slicks in rain is catastrophic
            if self.rainfall and self.compound in ["SOFT", "MEDIUM", "HARD"]:
                reward -= 15.0   # MUST pit for inters/wets in rain
                
            # 7. Track temperature degradation
            #    High track temp accelerates degradation, especially on softs
            if self.track_temp > 40.0 and self.compound == "SOFT" and self.tyre_age > opt_lo:
                reward -= 3.0    # Hot track + worn softs = bad
                
            # 8. Speed trap context: if speed is unusually low, car is struggling
            if self.speed_st < 200.0 and self.tyre_age > 10:
                reward -= 2.0    # Car performance suffering
                
        # ──────────────────────────────────────────────────────────────────────
        # ADVANCE TO NEXT LAP
        # ──────────────────────────────────────────────────────────────────────
        self.current_lap_idx += 1
        
        if self.current_lap_idx < len(self.current_laps):
            # Normal sync
            self._sync_state_from_data()
            
            # If agent chose to stay out, but the real driver pitted, simulate continuing
            real_lap = self.current_laps.iloc[self.current_lap_idx]
            if real_lap.get("is_pit_in", False) and action in [2, 3]:
                self.tyre_age += 1
                self.lap_time_delta += 0.15  # simulate continuing degradation
                reward -= 5.0    # Missed real pit window
            
            # If agent chose to pit, simulate fresh tyres
            if action in [0, 1]:
                self.tyre_age = 1
                self.lap_time_delta = 0.0  # fresh tyres
            
            terminated = False
        else:
            terminated = True
            
        truncated = False
        
        if terminated:
            # End of race rewards
            if self.position <= 10:
                reward += 30.0   # Points finish!
            if self.position <= 3:
                reward += 20.0   # Podium bonus
            if self.num_pits > 3:
                reward -= 30.0   # Too many pits
                
        return self._get_obs(), float(reward), terminated, truncated, {}


if __name__ == "__main__":
    os.makedirs("../checkpoints", exist_ok=True)
    
    print("=" * 60)
    print("StratoRace — Multi-Feature PPO Training v2")
    print("=" * 60)
    print("Initializing Data-Driven F1 Pit Strategy Environment...")
    env = F1PitStrategyEnv()
    eval_env = F1PitStrategyEnv()
    
    # Callback to save convergence data
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="../checkpoints/",
        log_path="../data/", 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    print("Creating new PPO model (11 features, multi-feature reward)...")
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,    # Encourage exploration
    )
    
    print("Training the agent on real FastF1 data (500k timesteps)...")
    model.learn(total_timesteps=500_000, callback=eval_callback)
    
    print("Saving final model...")
    model.save("../checkpoints/ppo_pit_strategy_final.zip")
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
