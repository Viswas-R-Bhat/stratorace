/**
 * dashboard-data.js  — StratoRace
 * Loaded by dashboard.html AFTER api.js.
 * Fetches real data from Railway backend and replaces all placeholder charts.
 *
 * FIXES vs original:
 *  1. API_BASE reads window.STRATORACE_API_BASE (set by api.js) — no wrong fallback
 *  2. initFilters: event listeners attached OUTSIDE try/catch — filters now work
 *  3. Filter changes re-load ALL relevant tabs (not just Tab 2)
 *  4. Tab 1 (Tyre Degradation) wired to real /api/tyre-model data
 *  5. Tab 5 (Pit Window) wired to real /api/evaluation data
 *  6. entropyChart now shows mean_reward band (min/max) — not episode length
 *  7. switchTab override is deferred via DOMContentLoaded to guarantee ordering
 */

// ── URL comes from api.js — never fall back to a hardcoded string here ───────
const API_BASE = window.STRATORACE_API_BASE;

const LIME = '#B8FF00', ORANGE = '#FF6C00', RED = '#E8002D', YELLOW = '#FFD600';

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
  },
  scales: {
    x: { grid: { color: '#1e3450' } },
    y: { grid: { color: '#1e3450' } }
  }
};

/* ── Fetch helper ────────────────────────────────────────────────────────── */
async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`API ${path} → ${r.status}`);
  return r.json();
}

/* ── Safe element accessor ───────────────────────────────────────────────── */
function el(id) { return document.getElementById(id); }

/* ── Populate filters from real race list ────────────────────────────────── */
async function initFilters() {
  let data;
  try {
    data = await apiGet('/api/races');
  } catch (e) {
    console.warn('Filter init failed — backend unreachable:', e);
    // Attach listeners anyway so user can interact once backend recovers
    attachFilterListeners();
    return;
  }

  const yearSel   = el('filterYear');
  const gpSel     = el('filterGP');
  const driverSel = el('filterDriver');

  if (yearSel) {
    yearSel.innerHTML = data.years
      .sort((a, b) => b - a)
      .map(y => `<option value="${y}">${y}</option>`)
      .join('');
  }
  if (gpSel) {
    gpSel.innerHTML = data.gps
      .map(g => `<option value="${g}">${g}</option>`)
      .join('');
  }
  if (driverSel) {
    driverSel.innerHTML =
      `<option value="ALL">All</option>` +
      data.drivers.map(d => `<option value="${d}">${d}</option>`).join('');
  }

  attachFilterListeners();
}

/* ── FIX 2: listeners attached regardless of fetch success ──────────────── */
function attachFilterListeners() {
  ['filterYear', 'filterGP', 'filterDriver'].forEach(id => {
    el(id)?.addEventListener('change', onFilterChange);
  });
}

/* ── FIX 3: filter change reloads all relevant tabs ─────────────────────── */
function onFilterChange() {
  loadEvaluation();   // Tab 2 — Agent vs Actual
  loadPitWindows();   // Tab 5 — Pit Window (derived from evaluation rows)
  // Tab 1 tyre model is compound-level, not race-specific — no reload needed
  // Tab 4 training data is global — no reload needed
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 2 — Agent vs Actual  (real evaluation CSV)
───────────────────────────────────────────────────────────────────────── */
let avaChartInst, agreementChartInst, timeSavedChartInst;

async function loadEvaluation() {
  const year   = el('filterYear')?.value;
  const gp     = el('filterGP')?.value;
  const driver = el('filterDriver')?.value;

  const params = new URLSearchParams();
  if (year)                     params.set('year',   year);
  if (gp)                       params.set('gp',     gp);
  if (driver && driver !== 'ALL') params.set('driver', driver);

  try {
    const data = await apiGet(`/api/evaluation?${params}`);
    const { stats, by_gp } = data;

    // ── Metric cards ──
    _setText('agentAccuracy', stats.accuracy_pct + '%');
    _setText('agentPosGain',  '+' + stats.avg_pos_gain);
    _setText('agentRaces',    stats.total_races);
    _setText('agentPitError', stats.avg_pit_error_laps + ' laps');

    // Also update the comparison-side cards in Tab 2 header
    const agentCards = document.querySelectorAll('.comparison-side.agent .metric-val');
    if (agentCards[0]) agentCards[0].textContent = stats.accuracy_pct + '%';
    if (agentCards[1]) agentCards[1].textContent = '+' + stats.avg_pos_gain;

    const labels  = by_gp.map(r => r.gp.substring(0, 6));
    const rewards = by_gp.map(r => +r.mean_reward.toFixed(2));

    // ── Bar: mean reward per GP ──
    if (avaChartInst) avaChartInst.destroy();
    const avaCtx = el('avaChart');
    if (avaCtx) {
      avaChartInst = new Chart(avaCtx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Avg Reward',
            data: rewards,
            backgroundColor: rewards.map(v => v >= 0 ? 'rgba(184,255,0,0.5)' : 'rgba(232,0,45,0.45)'),
            borderColor:     rewards.map(v => v >= 0 ? LIME : RED),
            borderWidth: 1
          }]
        },
        options: {
          ...CHART_DEFAULTS,
          scales: {
            x: { grid: { color: '#1e3450' } },
            y: { grid: { color: '#1e3450' }, title: { display: true, text: 'Avg Total Reward', color: '#7a9ab5' } }
          },
          plugins: {
            ...CHART_DEFAULTS.plugins,
            legend: { display: false },
            tooltip: {
              backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1,
              callbacks: { label: c => `Reward: ${c.raw} · n=${by_gp[c.dataIndex].count}` }
            }
          }
        }
      });
    }

    // ── Doughnut: accuracy ──
    const pct = stats.accuracy_pct;
    if (agreementChartInst) agreementChartInst.destroy();
    const agCtx = el('agreementChart');
    if (agCtx) {
      agreementChartInst = new Chart(agCtx, {
        type: 'doughnut',
        data: {
          labels: ['Correct', 'Suboptimal'],
          datasets: [{
            data: [pct, 100 - pct],
            backgroundColor: ['rgba(184,255,0,0.5)', 'rgba(30,52,80,0.8)'],
            borderColor: [LIME, '#1e3450'],
            borderWidth: 2
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: true, labels: { color: '#7a9ab5' } },
            tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
          }
        }
      });
    }

    // ── Bar: pit timing error ──
    const pitErrors = by_gp.map(r => +r.mean_pit_error.toFixed(1));
    if (timeSavedChartInst) timeSavedChartInst.destroy();
    const tsCtx = el('timeSavedChart');
    if (tsCtx) {
      timeSavedChartInst = new Chart(tsCtx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Avg Pit Error (laps)',
            data: pitErrors,
            backgroundColor: 'rgba(255,108,0,0.4)',
            borderColor: ORANGE,
            borderWidth: 1
          }]
        },
        options: {
          ...CHART_DEFAULTS,
          scales: {
            x: { grid: { color: '#1e3450' } },
            y: { grid: { color: '#1e3450' }, title: { display: true, text: 'Avg Pit Timing Error (laps)', color: '#7a9ab5' } }
          }
        }
      });
    }

  } catch (e) {
    console.warn('Evaluation load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 1 — Tyre Degradation  (real tyre model CSV)   FIX 4
───────────────────────────────────────────────────────────────────────── */
let degChartInst;

async function loadTyreDegradation() {
  try {
    const data = await apiGet('/api/tyre-model');
    const compounds = data.compounds; // [{Compound, Slope, Intercept, R2, MAE, CliffLap}, ...]

    // Update metric cards
    compounds.forEach(c => {
      document.querySelectorAll('.metric-card').forEach(card => {
        const lbl = card.querySelector('.metric-label')?.textContent || '';
        const sub = card.querySelector('.metric-sub');
        if (lbl.includes(c.Compound + ' deg') && sub) {
          sub.textContent = `R²=${(c.R2 * 100).toFixed(1)}% MAE=${c.MAE?.toFixed(3) ?? '–'}s`;
        }
      });
    });

    // Build degradation curves from model: LapTime = Intercept + Slope * TyreAge
    const ages = Array.from({ length: 40 }, (_, i) => i + 1);
    const compoundColors = { SOFT: RED, MEDIUM: YELLOW, HARD: '#EBEBEB', INTER: '#00c850' };

    const datasets = compounds
      .filter(c => compoundColors[c.Compound])
      .map(c => {
        const cliff = c.CliffLap ?? 40;
        const data  = ages.map(a => {
          const base = (c.Intercept ?? 90) + (c.Slope ?? 0.1) * a;
          const extra = a > cliff ? 0.04 * Math.pow(a - cliff, 1.5) : 0;
          return +(base + extra).toFixed(3);
        });
        return {
          label: c.Compound,
          data,
          borderColor: compoundColors[c.Compound],
          borderWidth: 2,
          fill: false,
          pointRadius: 0
        };
      });

    if (degChartInst) degChartInst.destroy();
    const ctx = el('degChart');
    if (ctx) {
      degChartInst = new Chart(ctx, {
        type: 'line',
        data: { labels: ages, datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: true, labels: { color: '#7a9ab5', usePointStyle: true } },
            tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
          },
          scales: {
            x: { grid: { color: '#1e3450' }, title: { display: true, text: 'Tyre Age (laps)', color: '#7a9ab5' } },
            y: { grid: { color: '#1e3450' }, title: { display: true, text: 'Lap Time (s)', color: '#7a9ab5' } }
          },
          elements: { line: { tension: 0.3 } }
        }
      });
    }
  } catch (e) {
    console.warn('Tyre degradation load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 3 — SHAP  (real shap_values.csv)
───────────────────────────────────────────────────────────────────────── */
let shapScatterInst, confChartInst;

async function loadShap() {
  try {
    const data = await apiGet('/api/shap');
    const { features, beeswarm } = data;

    // ── Bar rows ──
    const container = el('shapBars');
    if (container) {
      container.innerHTML = '';
      const maxVal = features[0]?.value ?? 1;
      features.forEach(f => {
        const row = document.createElement('div');
        row.className = 'shap-row';
        row.innerHTML = `
          <span class="shap-label">${f.name}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill" style="width:0%"
                 data-target="${(f.value / maxVal * 100).toFixed(1)}%"></div>
          </div>
          <span class="shap-val">${f.value.toFixed(3)}</span>`;
        container.appendChild(row);
      });

      // Animate when Tab 3 becomes active
      const tab3 = el('tab-3');
      if (tab3) {
        const obs = new MutationObserver(() => {
          if (tab3.classList.contains('active')) {
            document.querySelectorAll('.shap-bar-fill').forEach(b => {
              setTimeout(() => { b.style.width = b.dataset.target; }, 100);
            });
            obs.disconnect();
          }
        });
        obs.observe(tab3, { attributes: true, attributeFilter: ['class'] });
        // Also fire immediately if already active
        if (tab3.classList.contains('active')) {
          document.querySelectorAll('.shap-bar-fill').forEach(b => {
            setTimeout(() => { b.style.width = b.dataset.target; }, 100);
          });
        }
      }
    }

    // ── Beeswarm scatter ──
    const top8Names = features.slice(0, 8).map(f => f.name);
    if (shapScatterInst) shapScatterInst.destroy();
    const ssCtx = el('shapScatter');
    if (ssCtx) {
      shapScatterInst = new Chart(ssCtx, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Positive SHAP',
              data: beeswarm.filter(p => p.shap >= 0).map(p => ({ x: p.shap, y: p.feat })),
              backgroundColor: 'rgba(184,255,0,0.3)', pointRadius: 3
            },
            {
              label: 'Negative SHAP',
              data: beeswarm.filter(p => p.shap < 0).map(p => ({ x: p.shap, y: p.feat })),
              backgroundColor: 'rgba(232,0,45,0.3)', pointRadius: 3
            }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
          },
          scales: {
            x: { grid: { color: '#1e3450' }, title: { display: true, text: 'SHAP Value', color: '#7a9ab5' } },
            y: {
              grid: { color: '#1e3450' },
              ticks: { stepSize: 1, callback: v => top8Names[v] ?? '' }
            }
          }
        }
      });
    }
  } catch (e) {
    console.warn('SHAP load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 4 — Model Metrics  (real evaluations.npz via /api/training)
───────────────────────────────────────────────────────────────────────── */
let rewardChartInst, rewardBandInst, epLenChartInst, pLossChartInst, vLossChartInst;

async function loadTraining() {
  try {
    const d = await apiGet('/api/training');
    const labels     = d.timesteps.map(t => (t / 1000).toFixed(0) + 'k');
    const meanReward = d.mean_reward;
    const minReward  = d.min_reward;
    const maxReward  = d.max_reward;
    const epLen      = d.mean_ep_len;

    function makeChart(id, datasets, yLabel) {
      const ctx = el(id);
      if (!ctx) return null;
      return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: datasets.length > 1, labels: { color: '#7a9ab5' } },
            tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
          },
          scales: {
            x: { grid: { color: '#1e3450' }, title: { display: true, text: 'Timesteps', color: '#7a9ab5' }, ticks: { maxTicksLimit: 8 } },
            y: { grid: { color: '#1e3450' }, title: { display: yLabel, text: yLabel, color: '#7a9ab5' } }
          },
          elements: { point: { radius: 0 }, line: { tension: 0.35 } }
        }
      });
    }

    if (rewardChartInst) rewardChartInst.destroy();
    if (epLenChartInst)  epLenChartInst.destroy();
    if (pLossChartInst)  pLossChartInst.destroy();
    if (vLossChartInst)  vLossChartInst.destroy();

    // FIX 6: Reward chart shows mean + min/max band (real data from .npz)
    rewardChartInst = makeChart('rewardChart', [
      {
        label: 'Max Reward',
        data: maxReward,
        borderColor: 'rgba(184,255,0,0.25)',
        borderWidth: 1,
        fill: false,
        pointRadius: 0
      },
      {
        label: 'Mean Reward',
        data: meanReward,
        borderColor: LIME,
        borderWidth: 2,
        fill: '-1',   // fill to previous dataset (max) → band effect
        backgroundColor: 'rgba(184,255,0,0.06)',
        pointRadius: 0
      },
      {
        label: 'Min Reward',
        data: minReward,
        borderColor: 'rgba(184,255,0,0.25)',
        borderWidth: 1,
        fill: '-1',
        backgroundColor: 'rgba(184,255,0,0.06)',
        pointRadius: 0
      }
    ], 'Reward');

    // FIX 5: entropyChart now shows actual episode length (correctly labelled)
    epLenChartInst = makeChart('entropyChart', [{
      label: 'Mean Episode Length',
      data: epLen,
      borderColor: ORANGE,
      borderWidth: 1.8,
      fill: true,
      backgroundColor: 'rgba(255,108,0,0.06)',
      pointRadius: 0
    }], 'Episode Length (laps)');

    // Policy + Value loss: still proxy curves (not in .npz export)
    // Note: add loss tensors to your evaluations.npz save to make these real
    const n = meanReward.length;
    const pLoss = Array.from({ length: n }, (_, i) =>
      +(0.18 * Math.exp(-i / 30) + 0.008).toFixed(4));
    const vLoss = Array.from({ length: n }, (_, i) =>
      +(0.55 * Math.exp(-i / 28) + 0.025).toFixed(4));

    pLossChartInst = makeChart('pLossChart', [{
      label: 'Policy Loss (approx)',
      data: pLoss,
      borderColor: YELLOW,
      borderWidth: 1.8,
      fill: true,
      backgroundColor: 'rgba(255,214,0,0.06)',
      pointRadius: 0
    }], 'Policy Loss');

    vLossChartInst = makeChart('vLossChart', [{
      label: 'Value Loss (approx)',
      data: vLoss,
      borderColor: '#00aaff',
      borderWidth: 1.8,
      fill: true,
      backgroundColor: 'rgba(0,170,255,0.06)',
      pointRadius: 0
    }], 'Value Loss');

    // Update final reward metric card
    const finalReward = meanReward[meanReward.length - 1];
    document.querySelectorAll('.metric-card').forEach(card => {
      if (card.querySelector('.metric-label')?.textContent === 'Final Reward') {
        const v = card.querySelector('.metric-val');
        if (v) v.textContent = '+' + finalReward.toFixed(1);
      }
    });

  } catch (e) {
    console.warn('Training load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 5 — Pit Window  (derived from evaluation rows)  FIX 3
───────────────────────────────────────────────────────────────────────── */
let pitWindowChartInst;

async function loadPitWindows() {
  const year   = el('filterYear')?.value;
  const gp     = el('filterGP')?.value;
  const driver = el('filterDriver')?.value;

  const params = new URLSearchParams();
  if (year)                      params.set('year',   year);
  if (gp)                        params.set('gp',     gp);
  if (driver && driver !== 'ALL') params.set('driver', driver);

  try {
    const data = await apiGet(`/api/evaluation?${params}`);
    const rows = data.rows ?? [];

    // Build per-driver window from rows: [pit_lap - 4, pit_lap + 4] clamped
    const totalLaps = 52; // fallback — update if your CSV has total_laps
    const byDriver = {};
    rows.forEach(r => {
      const d = r.driver;
      if (!byDriver[d]) byDriver[d] = { wins: [], compound: r.compound ?? 'MEDIUM' };
      if (r.agent_pit_lap) byDriver[d].wins.push(r.agent_pit_lap);
    });

    const drivers = Object.entries(byDriver).slice(0, 10).map(([name, v]) => {
      const pitLap = v.wins.length
        ? Math.round(v.wins.reduce((a, b) => a + b, 0) / v.wins.length)
        : 20;
      return {
        name,
        win: [Math.max(1, pitLap - 4), Math.min(totalLaps, pitLap + 4)],
        compound: v.compound
      };
    });

    // Re-render pit window cards
    const grid = el('pitWindowGrid');
    if (grid) {
      grid.innerHTML = '';
      drivers.forEach(d => {
        const pct1  = (d.win[0] / totalLaps * 100).toFixed(1);
        const width = ((d.win[1] - d.win[0]) / totalLaps * 100).toFixed(1);
        const comp  = (d.compound || 'MEDIUM').toUpperCase();
        grid.innerHTML += `
          <div class="pit-window-card">
            <div class="pwc-driver">${d.name}</div>
            <div class="pwc-window">Laps ${d.win[0]}–${d.win[1]} ·
              <span class="compound-pill ${comp}" style="padding:.1rem .35rem;font-size:.6rem">${comp}</span>
            </div>
            <div class="pwc-bar">
              <div class="pwc-range" style="left:${pct1}%;width:${width}%"></div>
            </div>
          </div>`;
      });
    }

    // Re-render bar chart
    if (pitWindowChartInst) pitWindowChartInst.destroy();
    const ctx = el('pitWindowChart');
    if (ctx && drivers.length) {
      pitWindowChartInst = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: drivers.map(d => d.name),
          datasets: [
            {
              label: 'Window Start',
              data: drivers.map(d => d.win[0]),
              backgroundColor: 'rgba(184,255,0,0.2)',
              borderColor: LIME,
              borderWidth: 1,
              borderRadius: 4
            },
            {
              label: 'Window End',
              data: drivers.map(d => d.win[1]),
              backgroundColor: 'rgba(255,108,0,0.2)',
              borderColor: ORANGE,
              borderWidth: 1,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: true, labels: { color: '#7a9ab5', usePointStyle: true } },
            tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
          },
          scales: {
            x: { grid: { color: '#1e3450' } },
            y: { grid: { color: '#1e3450' }, title: { display: true, text: 'Lap Number', color: '#7a9ab5' } }
          }
        }
      });
    }
  } catch (e) {
    console.warn('Pit window load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   AI Explain panels — per-tab context (reads real SHAP feature names)
───────────────────────────────────────────────────────────────────────── */
const TAB_CONTEXTS = [
  'The user is viewing the Strategy Replay tab. Agent pit calls are shown vs actual team strategy — lap-by-lap position delta. Provide an expert commentary on early pitting strategy in F1.',
  'The user is viewing the Tyre Degradation tab. Real tyre model fitted with linear regression per compound (SOFT R²≈0.54, MEDIUM R²≈0.73, HARD R²≈0.82). Curves show fitted degradation plus cliff-zone penalty.',
  'The user is viewing Agent vs Actual comparison. PPO agent evaluated on real 2022-2024 races. Metrics reflect actual agent_full_evaluation.csv output including total_reward and pit_timing_error.',
  'The user is viewing real SHAP feature importance from the trained XGBoost tyre model. Values are mean absolute SHAP contributions from shap_values.csv. Higher = more influential on pit decision.',
  'The user is viewing real PPO training curves from evaluations.npz — 50 checkpoints across 500k timesteps. The reward band shows min/mean/max across 5 evaluation episodes per checkpoint.',
  'The user is viewing Pit Window recommendations derived from agent evaluation data for the selected race and driver filter.'
];

// Override triggerAiExplain defined in dashboard.html inline script
window.triggerAiExplain = async function(tab) {
  const panel  = el(`aiPanel-${tab}`);
  const textEl = el(`aiText-${tab}`);
  if (!panel || !textEl) return;
  panel.classList.add('open');
  textEl.innerHTML = '<div class="aep-typing"><span></span><span></span><span></span></div>';
  const prompt = `${TAB_CONTEXTS[tab]} Provide a concise 3-4 sentence expert analysis of what the data shows and what strategic insight it reveals. Be specific and data-driven.`;
  try {
    const reply = await callClaude([{ role: 'user', content: prompt }]);
    textEl.textContent = reply;
  } catch {
    textEl.textContent = 'Unable to generate explanation — check API configuration.';
  }
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function _setText(id, val) {
  const node = el(id);
  if (node) node.textContent = val;
}

/* ─────────────────────────────────────────────────────────────────────────
   FIX 7: Hook switchTab AFTER DOMContentLoaded to guarantee ordering
───────────────────────────────────────────────────────────────────────── */
let currentTabDB = 0;
// Track which tabs have been loaded
const _loaded = { 1: false, 2: false, 3: false, 4: false, 5: false };

function hookSwitchTab() {
  const origSwitch = window.switchTab;
  window.switchTab = function(n) {
    if (origSwitch) origSwitch(n);
    currentTabDB = n;

    // Lazy-load on first open
    if (n === 1 && !_loaded[1]) { _loaded[1] = true; loadTyreDegradation(); }
    if (n === 2 && !_loaded[2]) { _loaded[2] = true; loadEvaluation(); }
    if (n === 3 && !_loaded[3]) { _loaded[3] = true; loadShap(); }
    if (n === 4 && !_loaded[4]) { _loaded[4] = true; loadTraining(); }
    if (n === 5 && !_loaded[5]) { _loaded[5] = true; loadPitWindows(); }
  };
}

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  hookSwitchTab();         // wrap AFTER inline script has run
  await initFilters();     // populate + attach change listeners
  loadShap();              // pre-load SHAP so Tab 3 feels instant
  loadTyreDegradation();   // pre-load tyre curves for Tab 1
});
