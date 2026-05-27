StratoRace: F1 Pit Strategy AI
1. Project Overview & The Problem
StratoRace is a Formula 1 pit strategy optimisation system built using Reinforcement Learning (RL). It aims to solve the problem of pit-stop timing: a single mistimed pit stop can cost a driver multiple grid positions. Human intuition often falls short against the speed of modern F1 racing, with a 34% human error rate (deviating from the optimal window) observed in 2022–2024 races.

2. Methodology & Pipeline
The project pipelines three seasons of F1 telemetry through feature engineering into a trained Proximal Policy Optimisation (PPO) agent.

Target Variable: LapTimeDelta (isolates tyre degradation from raw car pace).
Baselines: Started with Linear Regression, Random Forest, and XGBoost (which reached 68% accuracy on pit timing).
PPO Agent: Actor-Critic MLP (FC256→FC128→FC64), trained in a custom OpenAI Gym environment. The agent makes live lap-by-lap decisions (PIT vs. STAY) rewarded by position gain and penalised by tyre cliff and pit time loss.
3. Data Sources
Coverage: 2022–2024 seasons (post-regulation era for stability). ~70 races, ~80k laps.
FastF1 API / F1 Timing: Laps, sectors, compounds, tyre age, position, gaps, pit in/out times.
Weather / Speed Traps: TrackTemp, Rainfall, speed traps (4 per lap).
Note: Safety car and VSC laps (outliers > 2x median) were removed to preserve tyre degradation signals.
4. Key Metrics & Performance
Validation Accuracy: The agent correctly identifies the optimal pit window 73.4% of the time on unseen races.
Agent vs Actual: Outperforms recorded human team strategy in 68% of decision comparisons.
Average Position Gain: +1.2 positions per race (compared to +0.4 for actual teams).
Top SHAP Feature: TyreAge, followed by LapTimeDelta and GapAhead.
Training: 12,400 episodes, final reward +284, policy loss converged to 0.012.
5. Technology Stack
Machine Learning & RL: PyTorch, PPO (Proximal Policy Optimisation), Custom Gym Environment, SHAP (Explainability).
Data Engineering: FastF1, XGBoost, Pandas.
Frontend & Backend: HTML/JS/CSS, FastAPI (Backend proxy), Anthropic Claude API (AI Assistant integration).
Deployment: Vercel (Frontend), Railway (Backend/API Proxy - currently being migrated).
6. Known Limitations
No Live Streaming Data: Trained on batch historical data; requires a live WebSocket feed for true real-time inference.
Binary Action Space: The agent currently only outputs {PIT, STAY}. Compound selection is handled by a heuristic.
Independent Drivers: No multi-agent team strategy modeling (e.g., stacking, deliberate undercuts).
No Safety Car Modeling: VSC/SC periods are excluded from training data.
Circuit-Agnostic: Does not explicitly encode circuit-specific degradation characteristics via embedding.
7. The Team
Dibyansh Raj: ML · RL Architecture · PPO Training
Sanskar Vishwas Raut: Data Engineering · Feature Pipeline
Viswas R Bhat: Frontend · Dashboard · AI Integration
