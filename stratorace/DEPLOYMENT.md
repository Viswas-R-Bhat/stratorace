# StratoRace — Deployment Guide
**Backend → Railway · Frontend → Vercel**

---

## Repository Structure

```
stratorace/
├── backend/
│   ├── main.py                    ← FastAPI app (all endpoints)
│   ├── requirements.txt
│   ├── Procfile                   ← Railway start command
│   ├── railway.json
│   └── data/
│       ├── shap_values.csv
│       ├── agent_full_evaluation.csv
│       ├── evaluations.npz
│       └── tyre_model_per_compound.csv
│
└── frontend/
    ├── index.html
    ├── dashboard.html
    ├── model.html
    ├── simulator.html
    ├── about.html
    ├── api.js                     ← Shared: API_BASE + callClaude proxy
    ├── dashboard-data.js          ← Real data for dashboard tabs
    ├── model-data.js              ← Real data for model page
    ├── stratorace-patch.js        ← Shared floating AI panel
    └── vercel.json
```

---

## Step 1 — Apply HTML patches (one-time, ~5 min)

Open each HTML file and make the following edits:

### All 5 HTML files — replace direct Anthropic fetch
Find every occurrence of:
```js
await fetch('https://api.anthropic.com/v1/messages', { ... })
```
And change the URL + response parsing to:
```js
// URL:
`${window.STRATORACE_API_BASE || 'https://YOUR-APP.up.railway.app'}/api/chat`
// Body: remove model/max_tokens, keep messages + system
body: JSON.stringify({ messages, system: SYSTEM_PROMPT })
// Response:
const data = await r.json(); return data.text || 'No response.';
```

### dashboard.html
Add before `</body>`:
```html
<script src="api.js"></script>
<script src="dashboard-data.js"></script>
```
Then **delete** or comment out in the `<script>` block:
- The `async function callClaude(...)` definition
- The `async function triggerAiExplain(tab)` definition
- The floating AI `sendFloatingMsg / addMsg / showTypingIndicator` functions
  (all replaced by dashboard-data.js)

### model.html
Add before `</body>`:
```html
<script src="api.js"></script>
<script src="model-data.js"></script>
```
Delete inline: `callClaude`, the SHAP bars `(function(){...})()` block,
and the `shapDist` + `featureInteraction` Chart initialisations.
(model-data.js loads real data and renders them.)

### about.html & index.html
Add before `</body>`:
```html
<script src="api.js"></script>
<script src="stratorace-patch.js"></script>
```
Delete inline `callClaude` definition. The `sendMessage / sendQa / sendAi`
functions can stay — just update the fetch URL and response parsing as above.

### simulator.html
Already patched — `simulator.html` in the frontend/ folder is ready to go.

---

## Step 2 — Deploy Backend to Railway

1. Push the `backend/` folder to a GitHub repo (can be a subfolder of a monorepo).
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo.
3. Set **Root Directory** to `backend/`.
4. Add environment variable:
   ```
   ANTHROPIC_API_KEY = sk-ant-...your key...
   ```
5. Railway auto-detects the `Procfile` and deploys.
6. Copy your Railway public URL, e.g.:
   ```
   https://stratorace-backend-production.up.railway.app
   ```

---

## Step 3 — Set Railway URL in frontend

In `api.js` (and `dashboard-data.js`, `model-data.js`, `simulator.html`), replace:
```js
const API_BASE = window.STRATORACE_API_BASE || 'https://stratorace-backend.up.railway.app';
```
with your actual Railway URL.

> **Tip:** To avoid editing 4 files, add a `config.js` with:
> ```js
> window.STRATORACE_API_BASE = 'https://YOUR-APP.up.railway.app';
> ```
> and load it first in every HTML file: `<script src="config.js"></script>`

---

## Step 4 — Deploy Frontend to Vercel

1. Push the `frontend/` folder to GitHub.
2. Go to [vercel.com](https://vercel.com) → New Project → Import repo.
3. Set **Root Directory** to `frontend/`.
4. Framework preset: **Other**.
5. No build command needed (pure static HTML).
6. Deploy. Vercel will use `vercel.json` for routing.

---

## API Endpoints Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Anthropic proxy — body: `{messages, system?}` |
| `POST` | `/api/simulate` | PPO simulator — body: `{compound, tyreAge, lap, totalLaps, position, gapAhead, gapBehind, trackTemp, rainfall}` |
| `GET`  | `/api/shap` | Real SHAP feature importance + beeswarm data |
| `GET`  | `/api/training` | PPO training curves from evaluations.npz |
| `GET`  | `/api/evaluation` | Agent evaluation (filter: year, gp, driver) |
| `GET`  | `/api/tyre-model` | Tyre model R², MAE, RMSE per compound |
| `GET`  | `/api/races` | Available years, GPs, drivers for filters |

---

## What Changed vs Original Frontend

| Page | Change |
|------|--------|
| All pages | Anthropic API key moved server-side (was exposed in browser JS) |
| `dashboard.html` | SHAP tab: real features (top = DeltaRolling3 0.23, not TyreAge) |
| `dashboard.html` | Model Metrics tab: real 500k-step training curves from evaluations.npz |
| `dashboard.html` | Agent vs Actual: real 100-race evaluation, 19 GPs, 22 drivers |
| `dashboard.html` | Filter dropdowns: populated from real data via `/api/races` |
| `model.html` | SHAP bars: real values, real beeswarm distribution |
| `model.html` | Feature interaction chart: real DeltaRolling3 vs StintLapDelta |
| `simulator.html` | Simulate button: calls `/api/simulate` (calibrated against real R²) |
| All pages | `triggerAiExplain()`: context includes real data facts |

---

## CORS

`main.py` sets `allow_origins=["*"]` for development.  
Before production, tighten it to your Vercel URL:
```python
allow_origins=["https://your-app.vercel.app"]
```

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-xxx uvicorn main:app --reload --port 8000

# Frontend — serve with any static server
cd frontend
python -m http.server 3000
# Then open http://localhost:3000

# Set local API_BASE in config.js:
# window.STRATORACE_API_BASE = 'http://localhost:8000';
```

Test all endpoints:
```bash
curl http://localhost:8000/api/shap | python -m json.tool | head -30
curl http://localhost:8000/api/training | python -m json.tool | head -20
curl http://localhost:8000/api/evaluation?year=2024 | python -m json.tool | head -30
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"compound":"SOFT","tyreAge":18,"lap":28,"totalLaps":52,"position":4,"gapAhead":3.0,"gapBehind":5.0,"trackTemp":38,"rainfall":false}'
```
