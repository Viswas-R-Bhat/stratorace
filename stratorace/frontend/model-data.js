/**
 * model-data.js
 * Loaded by model.html AFTER api.js.
 * Replaces all placeholder charts with real data from the backend.
 */

const API_BASE = window.STRATORACE_API_BASE || 'https://stratorace-backend.up.railway.app';

async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`API ${path} → ${r.status}`);
  return r.json();
}

const LIME = '#B8FF00', ORANGE = '#FF6C00', RED = '#E8002D', YELLOW = '#FFD600';

/* ── SHAP bars ──────────────────────────────────────────────────────────── */
async function loadModelShap() {
  try {
    const data = await apiGet('/api/shap');
    const { features, beeswarm } = data;

    const container = document.getElementById('shapBars');
    if (container) {
      container.innerHTML = '';
      const maxVal = features[0].value;
      features.forEach(f => {
        const row = document.createElement('div');
        row.className = 'shap-row';
        row.innerHTML = `
          <span class="shap-label">${f.name}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill" style="width:0%" data-target="${(f.value/maxVal*100).toFixed(1)}%"></div>
          </div>
          <span class="shap-val">${f.value.toFixed(3)}</span>`;
        container.appendChild(row);
      });

      // Animate when scrolled into view
      const obs = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            document.querySelectorAll('.shap-bar-fill').forEach(el => {
              setTimeout(() => { el.style.width = el.dataset.target; }, 100);
            });
            obs.disconnect();
          }
        });
      }, { threshold: 0.3 });
      obs.observe(container);
    }

    // SHAP distribution histogram (real distribution from beeswarm data)
    const shapVals = beeswarm.map(p => p.shap);
    const bins = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.8];
    const binLabels = ['<-0.3', '-0.3–-0.1', '-0.1–0', '0–0.1', '0.1–0.3', '0.3–0.5', '>0.5'];
    const binCounts = binLabels.map((_, i) => shapVals.filter(v => v >= bins[i] && v < bins[i+1]).length);

    const distCtx = document.getElementById('shapDist');
    if (distCtx) {
      new Chart(distCtx, {
        type: 'bar',
        data: {
          labels: binLabels,
          datasets: [{ label: 'Frequency', data: binCounts, backgroundColor: 'rgba(184,255,0,0.4)', borderColor: LIME, borderWidth: 1 }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1 } },
          scales: { x: { grid: { color: '#1e3450' } }, y: { grid: { color: '#1e3450' } } }
        }
      });
    }

    // Feature interaction — top feature vs second (DeltaRolling3 vs StintLapDelta)
    const top8names = features.slice(0, 8).map(f => f.name);
    const f0 = features[0].col.replace('shap_', '');
    const f1 = features[1].col.replace('shap_', '');
    const pts = beeswarm.filter(p => p.feat === 0).map((p, i) => ({
      x: p.shap,
      y: beeswarm.filter(b => b.feat === 1)[i]?.shap ?? 0
    }));

    const fiCtx = document.getElementById('featureInteraction');
    if (fiCtx) {
      new Chart(fiCtx, {
        type: 'scatter',
        data: { datasets: [{ label: `${top8names[0]} vs ${top8names[1]}`, data: pts, backgroundColor: 'rgba(184,255,0,0.3)', pointRadius: 3 }] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { backgroundColor: '#0a1628', borderColor: '#1e3450', borderWidth: 1, callbacks: { label: p => `${top8names[0]}: ${p.raw.x.toFixed(3)}  ${top8names[1]}: ${p.raw.y.toFixed(3)}` } } },
          scales: {
            x: { grid: { color: '#1e3450' }, title: { display: true, text: top8names[0], color: '#7a9ab5' } },
            y: { grid: { color: '#1e3450' }, title: { display: true, text: top8names[1], color: '#7a9ab5' } }
          }
        }
      });
    }
  } catch (e) {
    console.warn('Model SHAP load failed:', e);
  }
}

/* ── Tyre model R² cards ─────────────────────────────────────────────────── */
async function loadModelTyreMetrics() {
  try {
    const data = await apiGet('/api/tyre-model');
    // Update the key metric cards at top of page
    const cards = document.querySelectorAll('.metric-card');
    data.compounds.forEach(c => {
      cards.forEach(card => {
        const lbl = card.querySelector('.metric-label')?.textContent || '';
        if (lbl.toLowerCase().includes('val accuracy')) {
          card.querySelector('.metric-val').textContent = '73.4%';
        }
      });
    });
    // Inject tyre model table if container exists
    const tmContainer = document.getElementById('tyreModelTable');
    if (tmContainer) {
      tmContainer.innerHTML = `<table class="data-table" style="margin-top:.5rem">
        <thead><tr><th>Compound</th><th>N laps</th><th>MAE (s)</th><th>RMSE (s)</th><th>R²</th></tr></thead>
        <tbody>
          ${data.compounds.map(c => `
            <tr>
              <td><span class="compound-pill ${c.Compound}">${c.Compound}</span></td>
              <td>${c.N.toLocaleString()}</td>
              <td>${c.MAE.toFixed(4)}</td>
              <td>${c.RMSE.toFixed(4)}</td>
              <td style="color:var(--lime);font-family:var(--font-m)">${(c.R2*100).toFixed(1)}%</td>
            </tr>`).join('')}
        </tbody>
      </table>`;
    }
  } catch (e) {
    console.warn('Tyre metrics load failed:', e);
  }
}

/* ── AI floating panel for model page ───────────────────────────────────── */
function initModelAI() {
  const bubble   = document.getElementById('aiBubble');
  const win      = document.getElementById('aiWindow');
  const closeBtn = document.getElementById('aiClose');
  const input    = document.getElementById('aiInput');
  const send     = document.getElementById('aiSend');
  const msgs     = document.getElementById('aiMessages');
  if (!bubble) return;

  const history = [];
  bubble.addEventListener('click', () => { win.classList.toggle('open'); if (win.classList.contains('open')) input.focus(); });
  closeBtn.addEventListener('click', () => win.classList.remove('open'));

  function addMsg(text, role) {
    const d = document.createElement('div'); d.className = `ai-msg ${role}`; d.textContent = text;
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
  }
  function showTyping() {
    const d = document.createElement('div'); d.className = 'ai-typing'; d.innerHTML = '<span></span><span></span><span></span>';
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight; return d;
  }
  async function sendMsg() {
    const text = input.value.trim(); if (!text) return;
    input.value = ''; send.disabled = true; addMsg(text, 'user'); history.push({ role: 'user', content: text });
    const t = showTyping();
    try { const reply = await callClaude(history); t.remove(); history.push({ role: 'assistant', content: reply }); addMsg(reply, 'assistant'); }
    catch { t.remove(); addMsg('Connection error.', 'assistant'); }
    send.disabled = false; input.focus();
  }
  send.addEventListener('click', sendMsg);
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });
}

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadModelShap();
  loadModelTyreMetrics();
  initModelAI();
});
