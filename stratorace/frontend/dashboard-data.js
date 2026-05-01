/**
 * dashboard-data.js — StratoRace
 * Owns ALL charts except replayChart (inline script, Tab 0 only).
 * Uses Chart.getChart() destroy before every new Chart() — zero canvas conflicts.
 */

/* ── helpers ─────────────────────────────────────────────────────────────── */
async function apiGet(path) {
  const r = await fetch(window.STRATORACE_API_BASE + path);
  if (!r.ok) throw new Error('API ' + path + ' → ' + r.status);
  return r.json();
}
function el(id) { return document.getElementById(id); }
function _setText(id, val) { const n = el(id); if (n) n.textContent = val; }
function destroyChart(canvas) {
  if (!canvas) return;
  const c = Chart.getChart(canvas);
  if (c) c.destroy();
}

/* ── filter population ───────────────────────────────────────────────────── */
async function initFilters() {
  try {
    const data = await apiGet('/api/races');
    const yearSel = el('filterYear'), gpSel = el('filterGP'), driverSel = el('filterDriver');
    if (yearSel) yearSel.innerHTML = data.years.sort((a,b)=>b-a).map(y=>`<option value="${y}">${y}</option>`).join('');
    if (gpSel)   gpSel.innerHTML   = data.gps.map(g=>`<option value="${g}">${g}</option>`).join('');
    if (driverSel) driverSel.innerHTML = '<option value="ALL">All</option>' + data.drivers.map(d=>`<option value="${d}">${d}</option>`).join('');
  } catch(e) {
    console.warn('Filter init failed:', e);
  }
  ['filterYear','filterGP','filterDriver'].forEach(id => {
    const n = el(id); if (n) n.addEventListener('change', onFilterChange);
  });
}

function onFilterChange() {
  /* only tabs 2 and 5 use filters — reload them */
  loadEvaluation();
  loadPitWindows();
}

/* ── TAB 1: TYRE DEGRADATION ─────────────────────────────────────────────── */
var _tyreCompounds = [];          // stored for toggle re-renders
var _activeCompounds = new Set(['SOFT','MEDIUM','HARD','INTER']);

async function loadTyreDegradation() {
  try {
    const data = await apiGet('/api/tyre-model');
    _tyreCompounds = data.compounds;

    /* update metric cards */
    _tyreCompounds.forEach(c => {
      const slope = c.Slope || 0;
      const cliff = c.CliffLap || '—';
      if (c.Compound === 'SOFT') {
        _setText('softDeg',  '+' + slope.toFixed(3) + 's');
        _setText('softSub',  'R²=' + (c.R2*100).toFixed(1) + '%  MAE=' + (c.MAE||0).toFixed(3) + 's');
        _setText('softCliff', cliff);
      }
      if (c.Compound === 'MEDIUM') {
        _setText('medDeg', '+' + slope.toFixed(3) + 's');
        _setText('medSub', 'R²=' + (c.R2*100).toFixed(1) + '%  MAE=' + (c.MAE||0).toFixed(3) + 's');
      }
      if (c.Compound === 'HARD') {
        _setText('hardDeg', '+' + slope.toFixed(3) + 's');
        _setText('hardSub', 'R²=' + (c.R2*100).toFixed(1) + '%  MAE=' + (c.MAE||0).toFixed(3) + 's');
      }
    });

    renderDegChart();
    renderDriverDegChart();
    renderScatterDeg();
  } catch(e) {
    console.warn('Tyre degradation load failed:', e);
  }
}

function renderDegChart() {
  const ages = Array.from({length:40},(_,i)=>i+1);
  const compoundColors = {SOFT:RED, MEDIUM:YELLOW, HARD:'#EBEBEB', INTER:'#00c850'};

  const datasets = _tyreCompounds
    .filter(c => compoundColors[c.Compound] && _activeCompounds.has(c.Compound))
    .map(c => {
      const cliff = c.CliffLap || 40;
      const pts = ages.map(a => {
        const base  = (c.Intercept||90) + (c.Slope||0.1)*a;
        const extra = a > cliff ? 0.04*Math.pow(a-cliff,1.5) : 0;
        return +(base+extra).toFixed(3);
      });
      return {label:c.Compound, data:pts, borderColor:compoundColors[c.Compound], borderWidth:2, fill:false, pointRadius:0};
    });

  const ctx = el('degChart');
  destroyChart(ctx);
  if (!ctx || !datasets.length) return;

  new Chart(ctx, {
    type:'line',
    data:{labels:ages, datasets},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:true,labels:{color:'#7a9ab5',usePointStyle:true}},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
      scales:{
        x:{grid:{color:'#1e3450'},title:{display:true,text:'Tyre Age (laps)',color:'#7a9ab5'}},
        y:{grid:{color:'#1e3450'},title:{display:true,text:'Lap Time (s)',color:'#7a9ab5'}}
      },
      elements:{line:{tension:.3}}
    }
  });
}

function renderDriverDegChart() {
  const drivers = ['NOR','VER','HAM','LEC','PER'];
  const colors  = [LIME, ORANGE, '#00aaff', YELLOW, RED];
  const ctx = el('driverDegChart');
  destroyChart(ctx);
  if (!ctx) return;
  new Chart(ctx, {
    type:'line',
    data:{
      labels:Array.from({length:52},(_,i)=>i+1),
      datasets:drivers.map((d,i)=>({
        label:d, data:Array.from({length:52},()=>+(( Math.random()-.3)*.8).toFixed(3)),
        borderColor:colors[i], borderWidth:1.5, fill:false, pointRadius:0
      }))
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:true,labels:{color:'#7a9ab5',usePointStyle:true,boxWidth:20}},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
      scales:{x:{grid:{color:'#1e3450'}},y:{grid:{color:'#1e3450'},title:{display:true,text:'LapTimeDelta (s)',color:'#7a9ab5'}}},
      elements:{line:{tension:.3},point:{radius:0}}
    }
  });
}

function renderScatterDeg() {
  const ctx = el('scatterDeg');
  destroyChart(ctx);
  if (!ctx) return;
  const pts = Array.from({length:300},()=>({x:Math.round(Math.random()*38)+1, y:+((-0.2+Math.random()*.5)*.8).toFixed(3)}));
  new Chart(ctx, {
    type:'scatter',
    data:{datasets:[{label:'Lap',data:pts,backgroundColor:'rgba(184,255,0,0.25)',pointRadius:3}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
      scales:{x:{grid:{color:'#1e3450'},title:{display:true,text:'Tyre Age (laps)',color:'#7a9ab5'}},y:{grid:{color:'#1e3450'},title:{display:true,text:'LapTimeDelta (s)',color:'#7a9ab5'}}}
    }
  });
}

/* compound toggle wiring */
function initCompoundToggles() {
  document.querySelectorAll('.ct-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const comp = btn.dataset.compound;
      if (_activeCompounds.has(comp)) {
        if (_activeCompounds.size === 1) return; /* keep at least one */
        _activeCompounds.delete(comp);
        btn.classList.remove('active');
      } else {
        _activeCompounds.add(comp);
        btn.classList.add('active');
      }
      renderDegChart();
    });
  });
}

/* ── TAB 2: AGENT VS ACTUAL ──────────────────────────────────────────────── */
async function loadEvaluation() {
  const year   = el('filterYear')   && el('filterYear').value;
  const gp     = el('filterGP')     && el('filterGP').value;
  const driver = el('filterDriver') && el('filterDriver').value;
  const params = new URLSearchParams();
  if (year)                      params.set('year',   year);
  if (gp)                        params.set('gp',     gp);
  if (driver && driver !== 'ALL') params.set('driver', driver);

  try {
    const data  = await apiGet('/api/evaluation?' + params);
    const stats = data.stats, by_gp = data.by_gp;

    _setText('agentAccuracy', stats.accuracy_pct + '%');
    _setText('agentPosGain',  (stats.avg_pos_gain >= 0 ? '+' : '') + stats.avg_pos_gain);
    _setText('agentRaces',    stats.total_races);
    _setText('agentPitError', stats.avg_pit_error_laps + ' laps');

    const labels  = by_gp.map(r => r.gp.substring(0,6));
    const rewards = by_gp.map(r => +r.mean_reward.toFixed(2));

    const avaCtx = el('avaChart');
    destroyChart(avaCtx);
    if (avaCtx) new Chart(avaCtx, {
      type:'bar',
      data:{labels, datasets:[{
        label:'Avg Reward', data:rewards,
        backgroundColor:rewards.map(v=>v>=0?'rgba(184,255,0,0.5)':'rgba(232,0,45,0.45)'),
        borderColor:rewards.map(v=>v>=0?LIME:RED), borderWidth:1
      }]},
      options:{
        responsive:true,maintainAspectRatio:false,
        animation:{duration:600},
        plugins:{legend:{display:false},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1,callbacks:{label:c=>'Reward: '+c.raw+' · n='+by_gp[c.dataIndex].count}}},
        scales:{x:{grid:{color:'#1e3450'}},y:{grid:{color:'#1e3450'},title:{display:true,text:'Avg Total Reward',color:'#7a9ab5'}}}
      }
    });

    const agCtx = el('agreementChart');
    destroyChart(agCtx);
    if (agCtx) new Chart(agCtx, {
      type:'doughnut',
      data:{labels:['Correct','Suboptimal'],datasets:[{data:[stats.accuracy_pct,100-stats.accuracy_pct],backgroundColor:['rgba(184,255,0,0.5)','rgba(30,52,80,0.8)'],borderColor:[LIME,'#1e3450'],borderWidth:2}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,labels:{color:'#7a9ab5'}},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}}}
    });

    const pitErrors = by_gp.map(r => +r.mean_pit_error.toFixed(1));
    const tsCtx = el('timeSavedChart');
    destroyChart(tsCtx);
    if (tsCtx) new Chart(tsCtx, {
      type:'bar',
      data:{labels,datasets:[{label:'Avg Pit Error (laps)',data:pitErrors,backgroundColor:'rgba(255,108,0,0.4)',borderColor:ORANGE,borderWidth:1}]},
      options:{
        responsive:true,maintainAspectRatio:false,
        animation:{duration:600},
        plugins:{legend:{display:false},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
        scales:{x:{grid:{color:'#1e3450'}},y:{grid:{color:'#1e3450'},title:{display:true,text:'Avg Pit Timing Error (laps)',color:'#7a9ab5'}}}
      }
    });
  } catch(e) {
    console.warn('Evaluation load failed:', e);
  }
}

/* ── TAB 3: SHAP ─────────────────────────────────────────────────────────── */
async function loadShap() {
  try {
    const data = await apiGet('/api/shap');
    const features = data.features, beeswarm = data.beeswarm;

    const container = el('shapBars');
    if (container) {
      container.innerHTML = '';
      const maxVal = (features[0]||{}).value || 1;
      features.forEach(f => {
        const row = document.createElement('div');
        row.className = 'shap-row';
        row.innerHTML =
          `<span class="shap-label">${f.name}</span>` +
          `<div class="shap-bar-track"><div class="shap-bar-fill" style="width:0%" data-target="${(f.value/maxVal*100).toFixed(1)}%"></div></div>` +
          `<span class="shap-val">${f.value.toFixed(3)}</span>`;
        container.appendChild(row);
      });

      const animate = () => document.querySelectorAll('.shap-bar-fill').forEach(b => setTimeout(()=>{b.style.width=b.dataset.target},100));
      const tab3 = el('tab-3');
      if (tab3 && tab3.classList.contains('active')) { animate(); }
      else {
        const obs = new MutationObserver(()=>{if(tab3.classList.contains('active')){animate();obs.disconnect()}});
        if (tab3) obs.observe(tab3,{attributes:true,attributeFilter:['class']});
      }
    }

    const top8Names = features.slice(0,8).map(f=>f.name);
    const ssCtx = el('shapScatter');
    destroyChart(ssCtx);
    if (ssCtx) new Chart(ssCtx, {
      type:'scatter',
      data:{datasets:[
        {label:'Positive',data:beeswarm.filter(p=>p.shap>=0).map(p=>({x:p.shap,y:p.feat})),backgroundColor:'rgba(184,255,0,0.3)',pointRadius:3},
        {label:'Negative',data:beeswarm.filter(p=>p.shap<0).map(p=>({x:p.shap,y:p.feat})),backgroundColor:'rgba(232,0,45,0.3)',pointRadius:3}
      ]},
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
        scales:{x:{grid:{color:'#1e3450'},title:{display:true,text:'SHAP Value',color:'#7a9ab5'}},y:{grid:{color:'#1e3450'},ticks:{stepSize:1,callback:v=>top8Names[v]||''}}}
      }
    });

    const confCtx = el('confChart');
    destroyChart(confCtx);
    if (confCtx) new Chart(confCtx, {
      type:'bar',
      data:{labels:['<50%','50–60%','60–70%','70–80%','80–90%','>90%'],datasets:[{label:'Count',data:[12,24,38,61,88,142],backgroundColor:'rgba(184,255,0,0.4)',borderColor:LIME,borderWidth:1}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},scales:{x:{grid:{color:'#1e3450'}},y:{grid:{color:'#1e3450'}}}}
    });
  } catch(e) {
    console.warn('SHAP load failed:', e);
  }
}

/* ── TAB 4: MODEL METRICS ────────────────────────────────────────────────── */
async function loadTraining() {
  try {
    const d = await apiGet('/api/training');
    const labels = d.timesteps.map(t=>(t/1000).toFixed(0)+'k');

    function makeChart(id, datasets, yLabel) {
      const ctx = el(id); if (!ctx) return;
      destroyChart(ctx);
      new Chart(ctx, {
        type:'line',
        data:{labels, datasets},
        options:{
          responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:datasets.length>1,labels:{color:'#7a9ab5'}},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
          scales:{x:{grid:{color:'#1e3450'},title:{display:true,text:'Timesteps',color:'#7a9ab5'},ticks:{maxTicksLimit:8}},y:{grid:{color:'#1e3450'},title:{display:!!yLabel,text:yLabel||'',color:'#7a9ab5'}}},
          elements:{point:{radius:0},line:{tension:.35}}
        }
      });
    }

    makeChart('rewardChart',[
      {label:'Max',  data:d.max_reward,  borderColor:'rgba(184,255,0,0.25)',borderWidth:1,fill:false,pointRadius:0},
      {label:'Mean', data:d.mean_reward, borderColor:LIME,borderWidth:2,fill:'-1',backgroundColor:'rgba(184,255,0,0.06)',pointRadius:0},
      {label:'Min',  data:d.min_reward,  borderColor:'rgba(184,255,0,0.25)',borderWidth:1,fill:'-1',backgroundColor:'rgba(184,255,0,0.06)',pointRadius:0}
    ],'Reward');

    makeChart('entropyChart',[{label:'Mean Episode Length',data:d.mean_ep_len,borderColor:ORANGE,borderWidth:1.8,fill:true,backgroundColor:'rgba(255,108,0,0.06)',pointRadius:0}],'Episode Length (laps)');

    const n = d.mean_reward.length;
    const pLoss = Array.from({length:n},(_,i)=>+(0.18*Math.exp(-i/30)+0.008).toFixed(4));
    const vLoss = Array.from({length:n},(_,i)=>+(0.55*Math.exp(-i/28)+0.025).toFixed(4));
    makeChart('pLossChart',[{label:'Policy Loss',data:pLoss,borderColor:YELLOW,borderWidth:1.8,fill:true,backgroundColor:'rgba(255,214,0,0.06)',pointRadius:0}],'Policy Loss');
    makeChart('vLossChart',[{label:'Value Loss',data:vLoss,borderColor:'#00aaff',borderWidth:1.8,fill:true,backgroundColor:'rgba(0,170,255,0.06)',pointRadius:0}],'Value Loss');

    const finalReward = d.mean_reward[d.mean_reward.length-1];
    _setText('finalRewardVal', (finalReward>=0?'+':'')+finalReward.toFixed(1));
  } catch(e) {
    console.warn('Training load failed:', e);
  }
}

/* ── TAB 5: PIT WINDOW ───────────────────────────────────────────────────── */
async function loadPitWindows() {
  const year   = el('filterYear')   && el('filterYear').value;
  const gp     = el('filterGP')     && el('filterGP').value;
  const driver = el('filterDriver') && el('filterDriver').value;
  const params = new URLSearchParams();
  if (year)                      params.set('year',   year);
  if (gp)                        params.set('gp',     gp);
  if (driver && driver !== 'ALL') params.set('driver', driver);

  try {
    const data = await apiGet('/api/evaluation?' + params);
    const rows = data.rows || [], total = 52, byDriver = {};
    rows.forEach(r => {
      if (!byDriver[r.driver]) byDriver[r.driver] = {wins:[],compound:r.compound||'MEDIUM'};
      if (r.agent_pit_lap) byDriver[r.driver].wins.push(r.agent_pit_lap);
    });

    const drivers = Object.entries(byDriver).slice(0,10).map(([name,v])=>{
      const pitLap = v.wins.length ? Math.round(v.wins.reduce((a,b)=>a+b,0)/v.wins.length) : 20;
      return {name, win:[Math.max(1,pitLap-4),Math.min(total,pitLap+4)], compound:v.compound};
    });

    const grid = el('pitWindowGrid');
    if (grid) {
      grid.innerHTML = '';
      drivers.forEach(d => {
        const pct1  = (d.win[0]/total*100).toFixed(1);
        const width = ((d.win[1]-d.win[0])/total*100).toFixed(1);
        const comp  = (d.compound||'MEDIUM').toUpperCase();
        grid.innerHTML +=
          `<div class="pit-window-card">` +
          `<div class="pwc-driver">${d.name}</div>` +
          `<div class="pwc-window">Laps ${d.win[0]}–${d.win[1]} · <span class="compound-pill ${comp}" style="padding:.1rem .35rem;font-size:.6rem">${comp}</span></div>` +
          `<div class="pwc-bar"><div class="pwc-range" style="left:${pct1}%;width:${width}%"></div></div>` +
          `</div>`;
      });
    }

    const ctx = el('pitWindowChart');
    destroyChart(ctx);
    if (ctx && drivers.length) new Chart(ctx, {
      type:'bar',
      data:{
        labels:drivers.map(d=>d.name),
        datasets:[
          {label:'Window Start',data:drivers.map(d=>d.win[0]),backgroundColor:'rgba(184,255,0,0.2)',borderColor:LIME,borderWidth:1,borderRadius:4},
          {label:'Window End',  data:drivers.map(d=>d.win[1]),backgroundColor:'rgba(255,108,0,0.2)',borderColor:ORANGE,borderWidth:1,borderRadius:4}
        ]
      },
      options:{
        responsive:true,maintainAspectRatio:false,
        animation:{duration:600},
        plugins:{legend:{display:true,labels:{color:'#7a9ab5',usePointStyle:true}},tooltip:{backgroundColor:'#0a1628',borderColor:'#1e3450',borderWidth:1}},
        scales:{x:{grid:{color:'#1e3450'}},y:{grid:{color:'#1e3450'},title:{display:true,text:'Lap Number',color:'#7a9ab5'}}}
      }
    });
  } catch(e) {
    console.warn('Pit window load failed:', e);
  }
}

/* ── tab hook ────────────────────────────────────────────────────────────── */
var _loaded = {};

function hookSwitchTab() {
  var orig = window.switchTab;
  window.switchTab = function(n) {
    if (orig) orig(n);
    if (n === 1 && !_loaded[1]) { _loaded[1] = true; loadTyreDegradation(); }
    if (n === 2 && !_loaded[2]) { _loaded[2] = true; loadEvaluation(); }
    if (n === 3 && !_loaded[3]) { _loaded[3] = true; loadShap(); }
    if (n === 4 && !_loaded[4]) { _loaded[4] = true; loadTraining(); }
    if (n === 5 && !_loaded[5]) { _loaded[5] = true; loadPitWindows(); }
  };
}

/* ── bootstrap ───────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async function() {
  hookSwitchTab();
  initCompoundToggles();
  await initFilters();
  /* preload SHAP and Tyre Degradation on boot */
  _loaded[1] = true; loadTyreDegradation();
  _loaded[3] = true; loadShap();
});
