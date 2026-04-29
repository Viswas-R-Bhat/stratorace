/**
 * StratoRace — shared API config
 * Single source of truth for the backend URL.
 * Loaded first on every page — sets window.STRATORACE_API_BASE so that
 * dashboard-data.js and simulator.html both inherit the correct value.
 */

// ── ONLY LINE YOU EVER NEED TO CHANGE ────────────────────────────────────────
const _RAILWAY_URL = 'https://stratorace-production.up.railway.app';
// ─────────────────────────────────────────────────────────────────────────────

// Expose globally so dashboard-data.js and simulator.html read the right URL
window.STRATORACE_API_BASE = _RAILWAY_URL;
const API_BASE = _RAILWAY_URL;

/* ── Anthropic proxy (keeps key server-side via Railway) ─────────────────── */
const SYSTEM_PROMPT = `You are the StratoRace AI assistant. StratoRace is an F1 pit strategy optimisation system built on a PPO reinforcement learning agent trained on 2022–2024 Formula 1 data (~70 races, ~80k laps).
You can answer questions about:
- The PPO model, reward function, training process
- Tyre degradation, pit strategy, F1 race strategy
- The data (FastF1, compounds, lap times, sectors)
- SHAP feature importance and what drives decisions
- Specific races, drivers, or results
- The StratoRace project and methodology
Keep answers concise (2-4 sentences). If the user asks anything outside this scope, respond: "I can only answer questions about the StratoRace project and F1 strategy model."`;

async function callClaude(messages, system = SYSTEM_PROMPT) {
  const r = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, system }),
  });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  const data = await r.json();
  return data.text || 'No response.';
}
