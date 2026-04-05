/* ═══════════════════════════════════════════════════════════════════════
   shared.js — Common UI components (navbar, theme) for both pages
   ═══════════════════════════════════════════════════════════════════════ */

/**
 * Inject the shared navbar into the page.
 * @param {'dashboard'|'configurator'} activePage — which page is current
 */
function initNavbar(activePage) {
  const nav = document.createElement('header');
  nav.className = 'app-header';
  nav.innerHTML = `
    <h1>
      ${activePage === 'configurator' ? '⚙️' : '📊'}
      <span>K8s GP Scheduler</span> —
      ${activePage === 'configurator' ? 'Configurator' : 'Results Dashboard'}
    </h1>
    <nav class="header-nav">
      <a href="/dashboard" class="nav-link ${activePage === 'dashboard' ? 'active' : ''}">📊 Dashboard</a>
      <a href="/configurator" class="nav-link ${activePage === 'configurator' ? 'active' : ''}">⚙ Configurator</a>
      <button class="theme-toggle" onclick="toggleTheme()" title="Schimbă tema">
        ${document.documentElement.getAttribute('data-theme') === 'dark' ? '☀️' : '🌙'} Temă
      </button>
    </nav>
  `;
  document.body.prepend(nav);
}

/* ── Theme ──────────────────────────────────────────────────────────── */
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  const btn = document.querySelector('.theme-toggle');
  if (btn) btn.textContent = (next === 'dark' ? '☀️' : '🌙') + ' Temă';
  localStorage.setItem('theme', next);
}

/* Apply saved theme immediately (called before DOM render via inline script) */
function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
}
