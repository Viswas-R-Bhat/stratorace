/**
 * stratorace-patch.js
 * Injected into every page AFTER api.js.
 * - Overrides the old inline callClaude with the proxy version from api.js.
 * - Provides initPageData() which pages call to load real backend data.
 */

/* ── Floating AI panel (shared across all pages) ─────────────────────── */
function initFloatingAI() {
  const bubble   = document.getElementById('aiBubble');
  const win      = document.getElementById('aiWindow');
  const closeBtn = document.getElementById('aiClose');
  const input    = document.getElementById('aiInput');
  const send     = document.getElementById('aiSend');
  const msgs     = document.getElementById('aiMessages');
  if (!bubble) return;

  const history = [];

  bubble.addEventListener('click', () => {
    win.classList.toggle('open');
    if (win.classList.contains('open')) input.focus();
  });
  closeBtn.addEventListener('click', () => win.classList.remove('open'));

  function addMsg(text, role) {
    const d = document.createElement('div');
    d.className = `ai-msg ${role}`;
    d.textContent = text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  function showTyping() {
    const d = document.createElement('div');
    d.className = 'ai-typing';
    d.innerHTML = '<span></span><span></span><span></span>';
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  const SYS = 'You are the StratoRace AI assistant. StratoRace is an F1 pit strategy system using a real PPO RL model. Answer concisely (2-4 sentences). Only discuss StratoRace, F1 strategy, or model decisions.';

  async function sendMsg() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    send.disabled = true;
    addMsg(text, 'user');
    history.push({ role: 'user', content: text });
    const t = showTyping();
    try {
      const reply = await callClaude(history, SYS);
      t.remove();
      history.push({ role: 'assistant', content: reply });
      addMsg(reply, 'assistant');
    } catch {
      t.remove();
      addMsg('Connection error — check API configuration.', 'assistant');
    }
    send.disabled = false;
    input.focus();
  }

  send.addEventListener('click', sendMsg);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
  });
}

document.addEventListener('DOMContentLoaded', initFloatingAI);
