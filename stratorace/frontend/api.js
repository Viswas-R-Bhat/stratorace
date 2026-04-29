/**
 * StratoRace — shared API config
 * Sets window.STRATORACE_API_BASE so every page uses the correct backend.
 * Deliberately avoids const declarations that conflict with inline scripts.
 */

// Single source of truth — only change this string to switch backends
window.STRATORACE_API_BASE = 'https://stratorace-production.up.railway.app';

/**
 * callClaude — proxies through Railway backend (key stays server-side).
 * Each page supplies its own SYSTEM_PROMPT via the system parameter.
 */
async function callClaude(messages, system) {
  const r = await fetch(window.STRATORACE_API_BASE + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, system }),
  });
  if (!r.ok) throw new Error('API error ' + r.status);
  const data = await r.json();
  return data.text || 'No response.';
}
