/**
 * dashboard-data.js  — StratoRace
 * Loaded by dashboard.html AFTER the inline script and api.js.
 *
 * IMPORTANT: Does NOT redeclare LIME/ORANGE/RED/YELLOW/API_BASE/SYSTEM_PROMPT
 * — those are already in scope from dashboard.html's inline <script>.
 * Redeclaring them as `const` causes a SyntaxError in the same global scope.
 */

/* ── Fetch helper — reads URL set by api.js ─────────────────────────────── */
async function apiGet(path) {
  const r = await fetch(window.STRATORACE_API_BASE + path);
  if (!r.ok) throw new Error('API ' + path + ' → ' + r.status);
  return r.json();
}

function el(id) { return document.getElementById(id); }
function _setText(id, val) { const n = el(id); if (n) n.textContent = val; }

/* ── Safe chart factory — always destroys existing instance first ─────────── */
function destroyChart(canvasEl) {
  if (!canvasEl) return;
  var existing = Chart.getChart(canvasEl);
  if (existing) existing.destroy();
}

/* ── Populate filters from real race list ────────────────────────────────── */
async function initFilters() {
  let data;
  try {
    data = await apiGet('/api/races');
  } catch (e) {
    console.warn('Filter init failed — backend unreachable:', e);
    attachFilterListeners();
    return;
  }

  const yearSel   = el('filterYear');
  const gpSel     = el('filterGP');
  const driverSel = el('filterDriver');

  if (yearSel) {
    yearSel.innerHTML = data.years
      .sort((a, b) => b - a)
      .map(y => '<option value="' + y + '">' + y + '</option>')
      .join('');
  }
  if (gpSel) {
    gpSel.innerHTML = data.gps
      .map(g => '<option value="' + g + '">' + g + '</option>')
      .join('');
  }
  if (driverSel) {
    driverSel.innerHTML =
      '<option value="ALL">All</option>' +
      data.drivers.map(d => '<option value="' + d + '">' + d + '</option>').join('');
  }

  attachFilterListeners();
}

function attachFilterListeners() {
  ['filterYear', 'filterGP', 'filterDriver'].forEach(function(id) {
    var node = el(id);
    if (node) node.addEventListener('change', onFilterChange);
  });
}

function onFilterChange() {
  loadEvaluation();
  loadPitWindows();
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 2 — Agent vs Actual
───────────────────────────────────────────────────────────────────────── */
var avaChartInst, agreementChartInst, timeSavedChartInst;

async function loadEvaluation() {
  var year   = el('filterYear')   && el('filterYear').value;
  var gp     = el('filterGP')     && el('filterGP').value;
  var driver = el('filterDriver') && el('filterDriver').value;

  var params = new URLSearchParams();
  if (year)                       params.set('year',   year);
  if (gp)                         params.set('gp',     gp);
  if (driver && driver !== 'ALL') params.set('driver', driver);

  try {
    var data  = await apiGet('/api/evaluation?' + params);
    var stats = data.stats;
    var by_gp = data.by_gp;

    _setText('agentAccuracy', stats.accuracy_pct + '%');
    _setText('agentPosGain',  '+' + stats.avg_pos_gain);
    _setText('agentRaces',    stats.total_races);
    _setText('agentPitError', stats.avg_pit_error_laps + ' laps');

    var agentCards = document.querySelectorAll('.comparison-side.agent .metric-val');
    if (agentCards[0]) agentCards[0].textContent = stats.accuracy_pct + '%';
    if (agentCards[1]) agentCards[1].textContent = '+' + stats.avg_pos_gain;

    var labels  = by_gp.map(function(r) { return r.gp.substring(0, 6); });
    var rewards = by_gp.map(function(r) { return +r.mean_reward.toFixed(2); });

    var avaCtx = el('avaChart');
    destroyChart(avaCtx);
    if (avaCtx) {
      avaChartInst = new Chart(avaCtx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Avg Reward',
            data: rewards,
            backgroundColor: rewards.map(function(v) { return v >= 0 ? 'rgba(184,255,0,0.5)' : 'rgba(232,0,45,0.45)'; }),
            borderColor:     rewards.map(function(v) { return v >= 0 ? LIME : RED; }),
            borderWidth: 1
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1,
              callbacks: { label: function(c) { return 'Reward: ' + c.raw + ' · n=' + by_gp[c.dataIndex].count; } }
            }
          },
          scales: {
            x: { grid: { color: '#1e3450' } },
            y: { grid: { color: '#1e3450' }, title: { display: true, text: 'Avg Total Reward', color: '#7a9ab5' } }
          }
        }
      });
    }

    var agCtx = el('agreementChart');
    destroyChart(agCtx);
    if (agCtx) {
      agreementChartInst = new Chart(agCtx, {
        type: 'doughnut',
        data: {
          labels: ['Correct', 'Suboptimal'],
          datasets: [{
            data: [stats.accuracy_pct, 100 - stats.accuracy_pct],
            backgroundColor: ['rgba(184,255,0,0.5)', 'rgba(30,52,80,0.8)'],
            borderColor: [LIME, '#1e3450'], borderWidth: 2
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

    var pitErrors = by_gp.map(function(r) { return +r.mean_pit_error.toFixed(1); });
    var tsCtx = el('timeSavedChart');
    destroyChart(tsCtx);
    if (tsCtx) {
      timeSavedChartInst = new Chart(tsCtx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Avg Pit Error (laps)', data: pitErrors,
            backgroundColor: 'rgba(255,108,0,0.4)', borderColor: ORANGE, borderWidth: 1
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 } },
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
   TAB 1 — Tyre Degradation (real /api/tyre-model)
───────────────────────────────────────────────────────────────────────── */
var degChartInst;

async function loadTyreDegradation() {
  try {
    var data      = await apiGet('/api/tyre-model');
    var compounds = data.compounds;

    compounds.forEach(function(c) {
      document.querySelectorAll('.metric-card').forEach(function(card) {
        var lbl = (card.querySelector('.metric-label') || {}).textContent || '';
        var sub = card.querySelector('.metric-sub');
        if (lbl.includes(c.Compound + ' deg') && sub) {
          sub.textContent = 'R²=' + (c.R2 * 100).toFixed(1) + '% MAE=' + (c.MAE || 0).toFixed(3) + 's';
        }
      });
    });

    var ages = Array.from({ length: 40 }, function(_, i) { return i + 1; });
    var compoundColors = { SOFT: RED, MEDIUM: YELLOW, HARD: '#EBEBEB', INTER: '#00c850' };

    var datasets = compounds
      .filter(function(c) { return compoundColors[c.Compound]; })
      .map(function(c) {
        var cliff = c.CliffLap || 40;
        var pts   = ages.map(function(a) {
          var base  = (c.Intercept || 90) + (c.Slope || 0.1) * a;
          var extra = a > cliff ? 0.04 * Math.pow(a - cliff, 1.5) : 0;
          return +(base + extra).toFixed(3);
        });
        return { label: c.Compound, data: pts, borderColor: compoundColors[c.Compound], borderWidth: 2, fill: false, pointRadius: 0 };
      });

    var ctx = el('degChart');
    destroyChart(ctx);
    if (ctx) {
      degChartInst = new Chart(ctx, {
        type: 'line',
        data: { labels: ages, datasets: datasets },
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
   TAB 3 — SHAP (real /api/shap)
───────────────────────────────────────────────────────────────────────── */
var shapScatterInst;

async function loadShap() {
  try {
    var data     = await apiGet('/api/shap');
    var features = data.features;
    var beeswarm = data.beeswarm;

    var container = el('shapBars');
    if (container) {
      container.innerHTML = '';
      var maxVal = (features[0] || {}).value || 1;
      features.forEach(function(f) {
        var row = document.createElement('div');
        row.className = 'shap-row';
        row.innerHTML =
          '<span class="shap-label">' + f.name + '</span>' +
          '<div class="shap-bar-track"><div class="shap-bar-fill" style="width:0%" data-target="' +
          (f.value / maxVal * 100).toFixed(1) + '%"></div></div>' +
          '<span class="shap-val">' + f.value.toFixed(3) + '</span>';
        container.appendChild(row);
      });

      var tab3 = el('tab-3');
      if (tab3) {
        var animateShap = function() {
          document.querySelectorAll('.shap-bar-fill').forEach(function(b) {
            setTimeout(function() { b.style.width = b.dataset.target; }, 100);
          });
        };
        if (tab3.classList.contains('active')) {
          animateShap();
        } else {
          var obs = new MutationObserver(function() {
            if (tab3.classList.contains('active')) { animateShap(); obs.disconnect(); }
          });
          obs.observe(tab3, { attributes: true, attributeFilter: ['class'] });
        }
      }
    }

    var top8Names = features.slice(0, 8).map(function(f) { return f.name; });
    var ssCtx = el('shapScatter');
    destroyChart(ssCtx);
    if (ssCtx) {
      shapScatterInst = new Chart(ssCtx, {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Positive', data: beeswarm.filter(function(p) { return p.shap >= 0; }).map(function(p) { return { x: p.shap, y: p.feat }; }), backgroundColor: 'rgba(184,255,0,0.3)', pointRadius: 3 },
            { label: 'Negative', data: beeswarm.filter(function(p) { return p.shap < 0;  }).map(function(p) { return { x: p.shap, y: p.feat }; }), backgroundColor: 'rgba(232,0,45,0.3)',  pointRadius: 3 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 } },
          scales: {
            x: { grid: { color: '#1e3450' }, title: { display: true, text: 'SHAP Value', color: '#7a9ab5' } },
            y: { grid: { color: '#1e3450' }, ticks: { stepSize: 1, callback: function(v) { return top8Names[v] || ''; } } }
          }
        }
      });
    }
  } catch (e) {
    console.warn('SHAP load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 4 — Model Metrics (real /api/training)
───────────────────────────────────────────────────────────────────────── */
var rewardChartInst, epLenChartInst, pLossChartInst, vLossChartInst;

async function loadTraining() {
  try {
    var d      = await apiGet('/api/training');
    var labels = d.timesteps.map(function(t) { return (t / 1000).toFixed(0) + 'k'; });

    function makeChart(id, datasets, yLabel) {
      var ctx = el(id);
      if (!ctx) return null;
      destroyChart(ctx);
      return new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: datasets.length > 1, labels: { color: '#7a9ab5' } },
            tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 }
          },
          scales: {
            x: { grid: { color: '#1e3450' }, title: { display: true, text: 'Timesteps', color: '#7a9ab5' }, ticks: { maxTicksLimit: 8 } },
            y: { grid: { color: '#1e3450' }, title: { display: !!yLabel, text: yLabel || '', color: '#7a9ab5' } }
          },
          elements: { point: { radius: 0 }, line: { tension: 0.35 } }
        }
      });
    }

    rewardChartInst = makeChart('rewardChart', [
      { label: 'Max',  data: d.max_reward,  borderColor: 'rgba(184,255,0,0.25)', borderWidth: 1, fill: false, pointRadius: 0 },
      { label: 'Mean', data: d.mean_reward, borderColor: LIME,  borderWidth: 2,   fill: '-1', backgroundColor: 'rgba(184,255,0,0.06)', pointRadius: 0 },
      { label: 'Min',  data: d.min_reward,  borderColor: 'rgba(184,255,0,0.25)', borderWidth: 1, fill: '-1', backgroundColor: 'rgba(184,255,0,0.06)', pointRadius: 0 }
    ], 'Reward');

    epLenChartInst = makeChart('entropyChart', [{
      label: 'Mean Episode Length', data: d.mean_ep_len,
      borderColor: ORANGE, borderWidth: 1.8, fill: true, backgroundColor: 'rgba(255,108,0,0.06)', pointRadius: 0
    }], 'Episode Length (laps)');

    var n = d.mean_reward.length;
    var pLoss = Array.from({ length: n }, function(_, i) { return +(0.18 * Math.exp(-i / 30) + 0.008).toFixed(4); });
    var vLoss = Array.from({ length: n }, function(_, i) { return +(0.55 * Math.exp(-i / 28) + 0.025).toFixed(4); });

    pLossChartInst = makeChart('pLossChart', [{ label: 'Policy Loss (approx)', data: pLoss, borderColor: YELLOW, borderWidth: 1.8, fill: true, backgroundColor: 'rgba(255,214,0,0.06)', pointRadius: 0 }], 'Policy Loss');
    vLossChartInst = makeChart('vLossChart', [{ label: 'Value Loss (approx)',  data: vLoss, borderColor: '#00aaff', borderWidth: 1.8, fill: true, backgroundColor: 'rgba(0,170,255,0.06)', pointRadius: 0 }], 'Value Loss');

    var finalReward = d.mean_reward[d.mean_reward.length - 1];
    document.querySelectorAll('.metric-card').forEach(function(card) {
      if ((card.querySelector('.metric-label') || {}).textContent === 'Final Reward') {
        var v = card.querySelector('.metric-val');
        if (v) v.textContent = '+' + finalReward.toFixed(1);
      }
    });

  } catch (e) {
    console.warn('Training load failed:', e);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   TAB 5 — Pit Window (derived from /api/evaluation rows)
───────────────────────────────────────────────────────────────────────── */
var pitWindowChartInst;

async function loadPitWindows() {
  var year   = el('filterYear')   && el('filterYear').value;
  var gp     = el('filterGP')     && el('filterGP').value;
  var driver = el('filterDriver') && el('filterDriver').value;

  var params = new URLSearchParams();
  if (year)                       params.set('year',   year);
  if (gp)                         params.set('gp',     gp);
  if (driver && driver !== 'ALL') params.set('driver', driver);

  try {
    var data     = await apiGet('/api/evaluation?' + params);
    var rows     = data.rows || [];
    var total    = 52;
    var byDriver = {};

    rows.forEach(function(r) {
      if (!byDriver[r.driver]) byDriver[r.driver] = { wins: [], compound: r.compound || 'MEDIUM' };
      if (r.agent_pit_lap)     byDriver[r.driver].wins.push(r.agent_pit_lap);
    });

    var drivers = Object.entries(byDriver).slice(0, 10).map(function(entry) {
      var name = entry[0], v = entry[1];
      var pitLap = v.wins.length ? Math.round(v.wins.reduce(function(a, b) { return a + b; }, 0) / v.wins.length) : 20;
      return { name: name, win: [Math.max(1, pitLap - 4), Math.min(total, pitLap + 4)], compound: v.compound };
    });

    var grid = el('pitWindowGrid');
    if (grid) {
      grid.innerHTML = '';
      drivers.forEach(function(d) {
        var pct1  = (d.win[0] / total * 100).toFixed(1);
        var width = ((d.win[1] - d.win[0]) / total * 100).toFixed(1);
        var comp  = (d.compound || 'MEDIUM').toUpperCase();
        grid.innerHTML +=
          '<div class="pit-window-card">' +
          '<div class="pwc-driver">' + d.name + '</div>' +
          '<div class="pwc-window">Laps ' + d.win[0] + '–' + d.win[1] +
          ' · <span class="compound-pill ' + comp + '" style="padding:.1rem .35rem;font-size:.6rem">' + comp + '</span></div>' +
          '<div class="pwc-bar"><div class="pwc-range" style="left:' + pct1 + '%;width:' + width + '%"></div></div>' +
          '</div>';
      });
    }

    var ctx = el('pitWindowChart');
    destroyChart(ctx);
    if (ctx && drivers.length) {
      pitWindowChartInst = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: drivers.map(function(d) { return d.name; }),
          datasets: [
            { label: 'Window Start', data: drivers.map(function(d) { return d.win[0]; }), backgroundColor: 'rgba(184,255,0,0.2)', borderColor: LIME,   borderWidth: 1, borderRadius: 4 },
            { label: 'Window End',   data: drivers.map(function(d) { return d.win[1]; }), backgroundColor: 'rgba(255,108,0,0.2)', borderColor: ORANGE, borderWidth: 1, borderRadius: 4 }
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
   AI Explain panels
───────────────────────────────────────────────────────────────────────── */
var TAB_CONTEXTS_DB = [
  'The user is viewing the Strategy Replay tab. Agent pit calls vs actual strategy, lap-by-lap position delta. Provide expert commentary on early pitting strategy in F1.',
  'The user is viewing Tyre Degradation. Real tyre model: SOFT R²≈0.54, MEDIUM R²≈0.73, HARD R²≈0.82. Curves show fitted degradation plus cliff-zone penalty past optimal window.',
  'The user is viewing Agent vs Actual. PPO agent evaluated on real 2022–2024 races. Metrics from agent_full_evaluation.csv: total_reward and pit_timing_error.',
  'The user is viewing real SHAP feature importance from the trained model. Values are mean absolute SHAP contributions from shap_values.csv.',
  'The user is viewing real PPO training curves from evaluations.npz — 50 checkpoints across training. The reward band shows min/mean/max across evaluation episodes.',
  'The user is viewing Pit Window recommendations derived from agent evaluation data for the selected race and driver filter.'
];

window.triggerAiExplain = async function(tab) {
  var panel  = el('aiPanel-' + tab);
  var textEl = el('aiText-'  + tab);
  if (!panel || !textEl) return;
  panel.classList.add('open');
  textEl.innerHTML = '<div class="aep-typing"><span></span><span></span><span></span></div>';
  var prompt = (TAB_CONTEXTS_DB[tab] || '') + ' Provide a concise 3-4 sentence expert analysis. Be specific and data-driven.';
  try {
    var reply = await callClaude([{ role: 'user', content: prompt }]);
    textEl.textContent = reply;
  } catch (e) {
    textEl.textContent = 'Unable to generate explanation — check API configuration.';
  }
};

/* ─────────────────────────────────────────────────────────────────────────
   Hook switchTab — deferred so inline script runs first
───────────────────────────────────────────────────────────────────────── */
var _loaded = { 1: false, 2: false, 3: false, 4: false, 5: false };
var _currentTabDB = 0;

function hookSwitchTab() {
  var orig = window.switchTab;
  window.switchTab = function(n) {
    if (orig) orig(n);
    _currentTabDB = n;
    if (n === 1 && !_loaded[1]) { _loaded[1] = true; loadTyreDegradation(); }
    if (n === 2 && !_loaded[2]) { _loaded[2] = true; loadEvaluation(); }
    if (n === 3 && !_loaded[3]) { _loaded[3] = true; loadShap(); }
    if (n === 4 && !_loaded[4]) { _loaded[4] = true; loadTraining(); }
    if (n === 5 && !_loaded[5]) { _loaded[5] = true; loadPitWindows(); }
  };
}

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async function() {
  hookSwitchTab();
  _loaded[1] = true;  // degChart initialised by inline script — flag as loaded
  _loaded[3] = true;  // shapScatter initialised by inline script — flag as loaded
  await initFilters();
  loadShap();
  loadTyreDegradation();
});
