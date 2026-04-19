<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudGuard — Credit Card Fraud Detection System</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #030712;
    --surface: #0d1117;
    --surface2: #111827;
    --border: #1f2937;
    --border-glow: #10b98133;
    --accent: #10b981;
    --accent2: #f59e0b;
    --accent3: #3b82f6;
    --danger: #ef4444;
    --text: #f9fafb;
    --muted: #6b7280;
    --muted2: #374151;
    --code-bg: #0a0f1a;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  html { scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    line-height: 1.7;
    overflow-x: hidden;
  }

  /* ── ANIMATED GRID BACKGROUND ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(16,185,129,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(16,185,129,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── SCAN LINE ── */
  body::after {
    content: '';
    position: fixed;
    top: -100%;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: scan 8s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
  }

  @keyframes scan {
    0% { top: -2px; }
    100% { top: 100vh; }
  }

  /* ── LAYOUT ── */
  .container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 2rem;
    position: relative;
    z-index: 1;
  }

  /* ── HERO ── */
  .hero {
    padding: 80px 0 60px;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }

  .hero-glow {
    position: absolute;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 400px;
    background: radial-gradient(ellipse, rgba(16,185,129,0.12) 0%, transparent 70%);
    pointer-events: none;
  }

  .badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 28px;
    animation: fadeUp 0.6s ease both;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid;
    letter-spacing: 0.02em;
  }

  .badge-green { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.3); color: #34d399; }
  .badge-blue  { background: rgba(59,130,246,0.1); border-color: rgba(59,130,246,0.3); color: #60a5fa; }
  .badge-amber { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); color: #fbbf24; }
  .badge-red   { background: rgba(239,68,68,0.1);  border-color: rgba(239,68,68,0.3);  color: #f87171; }
  .badge-purple{ background: rgba(139,92,246,0.1); border-color: rgba(139,92,246,0.3); color: #a78bfa; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    animation: fadeUp 0.6s 0.1s ease both;
  }

  .hero-title .accent { color: var(--accent); }
  .hero-title .dim { color: var(--muted); }

  .hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 24px;
    animation: fadeUp 0.6s 0.2s ease both;
  }

  .hero-desc {
    max-width: 680px;
    color: #9ca3af;
    font-size: 16px;
    line-height: 1.75;
    margin-bottom: 36px;
    animation: fadeUp 0.6s 0.3s ease both;
  }

  .hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    animation: fadeUp 0.6s 0.4s ease both;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.2s ease;
    cursor: pointer;
    border: 1px solid;
  }

  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #000;
    box-shadow: 0 0 20px rgba(16,185,129,0.3);
  }
  .btn-primary:hover {
    background: #34d399;
    box-shadow: 0 0 32px rgba(16,185,129,0.5);
    transform: translateY(-1px);
  }

  .btn-secondary {
    background: transparent;
    border-color: var(--border);
    color: var(--muted);
  }
  .btn-secondary:hover {
    border-color: var(--accent);
    color: var(--accent);
    transform: translateY(-1px);
  }

  /* ── KPI STRIP ── */
  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin: 48px 0;
  }

  .kpi-item {
    background: var(--surface);
    padding: 24px;
    text-align: center;
    position: relative;
    transition: background 0.2s;
  }
  .kpi-item:hover { background: var(--surface2); }
  .kpi-item::after {
    content: '';
    position: absolute;
    bottom: 0; left: 50%; transform: translateX(-50%);
    width: 0; height: 2px;
    background: var(--accent);
    transition: width 0.3s ease;
  }
  .kpi-item:hover::after { width: 60%; }

  .kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 6px;
  }

  .kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* ── SECTION ── */
  section {
    padding: 64px 0;
    border-bottom: 1px solid var(--border);
  }

  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-label::before {
    content: '';
    display: block;
    width: 24px; height: 1px;
    background: var(--accent);
  }

  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    font-weight: 800;
    margin-bottom: 16px;
    letter-spacing: -0.01em;
  }

  .section-desc {
    color: #9ca3af;
    max-width: 640px;
    margin-bottom: 40px;
  }

  /* ── DASHBOARD CARD ── */
  .dashboard-hero-card {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: all 0.3s ease;
    cursor: pointer;
  }
  .dashboard-hero-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent), 0 0 40px rgba(16,185,129,0.15);
    transform: translateY(-3px);
  }

  .dashboard-hero-card img {
    width: 100%;
    display: block;
    transition: transform 0.4s ease;
  }
  .dashboard-hero-card:hover img { transform: scale(1.02); }

  .dashboard-hero-card .card-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(3,7,18,0.9) 0%, transparent 50%);
    display: flex;
    align-items: flex-end;
    padding: 28px;
    opacity: 0;
    transition: opacity 0.3s ease;
  }
  .dashboard-hero-card:hover .card-overlay { opacity: 1; }

  .card-overlay-content {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .overlay-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    background: var(--accent);
    color: #000;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    transition: background 0.2s;
  }
  .overlay-btn:hover { background: #34d399; }

  /* ── SCREENSHOT GRID ── */
  .screenshot-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 24px;
  }

  .screenshot-card {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: all 0.3s ease;
    group: true;
  }
  .screenshot-card:hover {
    border-color: rgba(16,185,129,0.4);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(16,185,129,0.2);
    transform: translateY(-4px);
    z-index: 2;
  }

  .screenshot-card img {
    width: 100%;
    display: block;
    transition: transform 0.4s ease;
  }
  .screenshot-card:hover img { transform: scale(1.04); }

  .screenshot-label {
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(3,7,18,0.85);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 9px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--accent);
    backdrop-filter: blur(6px);
  }

  .screenshot-caption {
    padding: 12px 14px;
    font-size: 12px;
    color: var(--muted);
    background: var(--surface2);
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── MODEL TABLE ── */
  .model-table-wrap {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }

  thead {
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }

  th {
    padding: 14px 18px;
    text-align: left;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }

  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--surface2); }
  tbody tr.active-row { background: rgba(16,185,129,0.05); }
  tbody tr.active-row td:first-child { color: var(--accent); }

  td { padding: 14px 18px; color: #d1d5db; }

  .metric-bar-wrap { display: flex; align-items: center; gap: 10px; }
  .metric-bar {
    flex: 1; height: 5px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }
  .metric-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
  }
  .fill-green { background: var(--accent); }
  .fill-blue  { background: var(--accent3); }
  .fill-amber { background: var(--accent2); }

  /* ── IMPACT CARDS ── */
  .impact-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }

  .impact-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
  }
  .impact-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .impact-card.green::before { background: var(--accent); }
  .impact-card.blue::before  { background: var(--accent3); }
  .impact-card.amber::before { background: var(--accent2); }
  .impact-card.red::before   { background: var(--danger); }

  .impact-card:hover {
    transform: translateY(-4px);
    border-color: rgba(16,185,129,0.3);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
  }

  .impact-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 6px;
  }
  .impact-card.green .impact-value { color: var(--accent); }
  .impact-card.blue .impact-value  { color: #60a5fa; }
  .impact-card.amber .impact-value { color: #fbbf24; }
  .impact-card.red .impact-value   { color: #f87171; }

  .impact-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* ── ARCH BLOCK ── */
  .arch-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .arch-col {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .arch-col:hover { border-color: rgba(16,185,129,0.3); }

  .arch-col-header {
    padding: 12px 18px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .arch-col pre {
    padding: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #9ca3af;
    line-height: 1.7;
    overflow-x: auto;
    white-space: pre;
  }

  /* ── CODE BLOCK ── */
  .code-block {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin: 24px 0;
  }

  .code-header {
    padding: 10px 18px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .code-lang {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .code-dots { display: flex; gap: 6px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .dot-red { background: #ef4444; }
  .dot-amber { background: #f59e0b; }
  .dot-green { background: #10b981; }

  .code-block pre {
    padding: 20px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    line-height: 1.7;
    color: #9ca3af;
    overflow-x: auto;
  }

  .code-block pre .kw  { color: #c084fc; }
  .code-block pre .fn  { color: #60a5fa; }
  .code-block pre .str { color: #34d399; }
  .code-block pre .cm  { color: #4b5563; }
  .code-block pre .num { color: #fbbf24; }
  .code-block pre .op  { color: #f472b6; }

  /* ── API TABLE ── */
  .api-table-wrap {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  .api-row {
    display: grid;
    grid-template-columns: 60px 200px 1fr;
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }
  .api-row:last-child { border-bottom: none; }
  .api-row:hover { background: var(--surface2); }

  .api-row.head {
    background: var(--surface2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .api-cell { padding: 14px 18px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #d1d5db; }

  .method {
    font-weight: 700;
    font-size: 11px;
  }
  .method.get  { color: #34d399; }
  .method.post { color: #60a5fa; }

  /* ── RISK TIER ── */
  .tier-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 12px;
    margin-top: 24px;
  }

  .tier-card {
    border-radius: 8px;
    padding: 20px 16px;
    border: 1px solid;
    transition: transform 0.2s;
  }
  .tier-card:hover { transform: translateY(-3px); }

  .tier-card.low    { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.25); }
  .tier-card.medium { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.25); }
  .tier-card.high   { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.25); }
  .tier-card.critical { background: rgba(139,92,246,0.08); border-color: rgba(139,92,246,0.25); }

  .tier-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
  }
  .tier-card.low .tier-name     { color: var(--accent); }
  .tier-card.medium .tier-name  { color: var(--accent2); }
  .tier-card.high .tier-name    { color: var(--danger); }
  .tier-card.critical .tier-name { color: #a78bfa; }

  .tier-range {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #f9fafb;
    margin-bottom: 4px;
  }
  .tier-action {
    font-size: 11px;
    color: var(--muted);
  }

  /* ── TECH STACK ── */
  .tech-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  .tech-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #d1d5db;
    transition: all 0.2s;
    cursor: default;
  }
  .tech-pill:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(16,185,129,0.05);
  }

  /* ── TOC NAV ── */
  .toc {
    position: fixed;
    top: 50%;
    right: 24px;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 100;
  }

  .toc-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted2);
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }
  .toc-dot:hover, .toc-dot.active {
    background: var(--accent);
    transform: scale(1.5);
  }
  .toc-dot::before {
    content: attr(data-label);
    position: absolute;
    right: 14px;
    top: 50%;
    transform: translateY(-50%);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text);
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
  }
  .toc-dot:hover::before { opacity: 1; }

  /* ── COLLAPSIBLE ── */
  details {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 12px;
    transition: border-color 0.2s;
  }
  details:hover { border-color: rgba(16,185,129,0.3); }
  details[open] { border-color: rgba(16,185,129,0.4); }

  summary {
    padding: 16px 20px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #d1d5db;
    background: var(--surface);
    display: flex;
    align-items: center;
    gap: 10px;
    list-style: none;
    transition: background 0.15s;
  }
  summary:hover { background: var(--surface2); }
  summary .arrow {
    margin-left: auto;
    color: var(--muted);
    transition: transform 0.2s;
  }
  details[open] summary .arrow { transform: rotate(90deg); }

  .details-body {
    padding: 20px;
    background: var(--code-bg);
    border-top: 1px solid var(--border);
  }
  .details-body pre {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #9ca3af;
    line-height: 1.8;
    white-space: pre-wrap;
  }

  /* ── FEATURE LIST ── */
  .feature-list {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .feature-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    transition: all 0.2s;
  }
  .feature-item:hover {
    border-color: rgba(16,185,129,0.3);
    background: var(--surface2);
    transform: translateX(4px);
  }

  .feature-icon {
    width: 32px; height: 32px;
    border-radius: 6px;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 14px;
  }

  .feature-text h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text);
    margin-bottom: 3px;
  }
  .feature-text p {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.5;
  }

  /* ── FOOTER ── */
  footer {
    padding: 40px 0;
    text-align: center;
    border-top: 1px solid var(--border);
  }

  footer .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--accent);
    margin-bottom: 8px;
  }

  footer p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }

  footer a { color: var(--accent); text-decoration: none; }
  footer a:hover { text-decoration: underline; }

  /* ── ANIMATIONS ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .reveal {
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  .reveal.visible { opacity: 1; transform: translateY(0); }

  /* ── RESPONSIVE ── */
  @media (max-width: 768px) {
    .kpi-strip     { grid-template-columns: repeat(2,1fr); }
    .impact-grid   { grid-template-columns: repeat(2,1fr); }
    .screenshot-grid { grid-template-columns: 1fr; }
    .arch-grid     { grid-template-columns: 1fr; }
    .tier-grid     { grid-template-columns: repeat(2,1fr); }
    .feature-list  { grid-template-columns: 1fr; }
    .toc           { display: none; }
  }

  /* ── CURSOR GLOW ── */
  .cursor-glow {
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%);
    position: fixed;
    pointer-events: none;
    z-index: 0;
    transform: translate(-50%, -50%);
    transition: opacity 0.3s;
  }

  /* ── DIVIDER ── */
  .divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 0;
  }
</style>
</head>
<body>

<!-- Cursor glow -->
<div class="cursor-glow" id="cursorGlow"></div>

<!-- TOC -->
<nav class="toc" id="toc">
  <div class="toc-dot active" data-label="Overview" data-target="hero" onclick="scrollTo('hero')"></div>
  <div class="toc-dot" data-label="Dashboard" data-target="dashboard" onclick="scrollTo('dashboard')"></div>
  <div class="toc-dot" data-label="Performance" data-target="performance" onclick="scrollTo('performance')"></div>
  <div class="toc-dot" data-label="Impact" data-target="impact" onclick="scrollTo('impact')"></div>
  <div class="toc-dot" data-label="Architecture" data-target="architecture" onclick="scrollTo('architecture')"></div>
  <div class="toc-dot" data-label="API" data-target="api" onclick="scrollTo('api')"></div>
  <div class="toc-dot" data-label="Setup" data-target="setup" onclick="scrollTo('setup')"></div>
</nav>

<!-- ═══════════════ HERO ═══════════════ -->
<section class="hero" id="hero">
  <div class="container">
    <div class="hero-glow"></div>

    <div class="badge-row">
      <span class="badge badge-green">⬡ Production-Ready</span>
      <span class="badge badge-blue">◈ XGBoost · PR-AUC 0.870</span>
      <span class="badge badge-amber">⚡ FastAPI · Streamlit</span>
      <span class="badge badge-purple">✦ MLflow · Optuna · SHAP</span>
      <span class="badge badge-red">⬟ 39 Tests Passing</span>
    </div>

    <h1 class="hero-title">
      Fraud<span class="accent">Guard</span><br>
      <span class="dim">ML Detection System</span>
    </h1>
    <p class="hero-sub">// credit card fraud detection · end-to-end · production-grade</p>
    <p class="hero-desc">
      A production-grade credit card fraud detection system built end-to-end — from raw transaction data
      to a live inference API, real-time analytics dashboard, experiment tracking, and automated drift monitoring.
      Not just a notebook. A real system.
    </p>

    <div class="hero-actions">
      <a href="https://fraud-detection-988itbtnyczqkfo3fqqk8e.streamlit.app/" target="_blank" class="btn btn-primary">
        ⬡ View Live Dashboard
      </a>
      <a href="https://github.com/RagannagariSiva/fraud-detection" target="_blank" class="btn btn-secondary">
        ◈ GitHub Repository
      </a>
    </div>

    <div class="kpi-strip reveal">
      <div class="kpi-item">
        <div class="kpi-value">0.870</div>
        <div class="kpi-label">PR-AUC Score</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">0.978</div>
        <div class="kpi-label">ROC-AUC Score</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">3,400%</div>
        <div class="kpi-label">ROI Estimate</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">39</div>
        <div class="kpi-label">Tests Passing</div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════ DASHBOARD ═══════════════ -->
<section id="dashboard">
  <div class="container">
    <div class="section-label">Live System</div>
    <h2 class="section-title reveal">Dashboard &amp; Screens</h2>
    <p class="section-desc reveal">Five-page Streamlit dashboard with real-time fraud alerts, model analysis, batch scoring, and live prediction — all connected to the FastAPI inference backend.</p>

    <!-- Main dashboard card -->
    <div class="dashboard-hero-card reveal">
      <img src="imgs/DashBord.png" alt="Live Dashboard" onerror="this.style.display='none'; this.parentElement.querySelector('.img-fallback').style.display='flex'">
      <div class="img-fallback" style="display:none; height:280px; align-items:center; justify-content:center; flex-direction:column; gap:12px; color:#4b5563;">
        <div style="font-size:3rem">📊</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:12px;">Dashboard Preview</div>
      </div>
      <div class="card-overlay">
        <div class="card-overlay-content">
          <a href="https://fraud-detection-988itbtnyczqkfo3fqqk8e.streamlit.app/" target="_blank" class="overlay-btn">
            ↗ Open Live Dashboard
          </a>
          <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#9ca3af;">Real-time · Auto-refresh · Live alerts</span>
        </div>
      </div>
    </div>

    <!-- Screenshot grid 2-col -->
    <div class="screenshot-grid reveal" style="margin-top:16px;">
      <div class="screenshot-card">
        <img src="imgs/Local%20host.png" alt="Live Prediction" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Page 02</div>
        <div class="screenshot-caption">⬡ Live Prediction — score any transaction with probability gauge</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Machine%20Learning.png" alt="Model Analysis" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Page 03</div>
        <div class="screenshot-caption">◈ Model Analysis — PR/ROC curves, confusion matrix, SHAP plots</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Screen%20short-1.png" alt="Screen 1" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Page 04</div>
        <div class="screenshot-caption">⬟ Alert Feed — streaming live fraud alerts from simulator</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Screen%20short-2.png" alt="Screen 2" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Page 05</div>
        <div class="screenshot-caption">⚡ Batch Scoring — upload CSV, get predictions for 10K rows</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Screen%20short-3.png" alt="Screen 3" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Detail</div>
        <div class="screenshot-caption">✦ Drift detection — PSI + KS across all feature distributions</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Screen%20short-4.png" alt="Screen 4" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Detail</div>
        <div class="screenshot-caption">◈ Rolling metrics — 5-min windows, P50 / P95 / P99 latency</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Screen%20short-5.png" alt="Screen 5" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Detail</div>
        <div class="screenshot-caption">⬡ Feature importance — SHAP values per transaction</div>
      </div>
      <div class="screenshot-card">
        <img src="imgs/Screen%20short-6.png" alt="Screen 6" onerror="this.style.background='#111827'; this.style.height='180px'">
        <div class="screenshot-label">Detail</div>
        <div class="screenshot-caption">⚡ Business impact — live cost/benefit estimates per threshold</div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════ PERFORMANCE ═══════════════ -->
<section id="performance">
  <div class="container">
    <div class="section-label">Evaluation</div>
    <h2 class="section-title reveal">Model Performance</h2>
    <p class="section-desc reveal">Evaluated on a held-out test set (20% of data, never seen during training or threshold calibration). All numbers from the real Kaggle dataset — not synthetic data.</p>

    <div class="model-table-wrap reveal">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>PR-AUC</th>
            <th>ROC-AUC</th>
            <th>Recall</th>
            <th>Precision</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          <tr class="active-row">
            <td>⬡ XGBoost <span style="background:rgba(16,185,129,0.15); color:var(--accent); font-size:10px; padding:2px 7px; border-radius:4px; margin-left:6px; font-family:'JetBrains Mono',monospace;">ACTIVE</span></td>
            <td><div class="metric-bar-wrap">0.870 <div class="metric-bar"><div class="metric-fill fill-green" style="width:87%"></div></div></div></td>
            <td><div class="metric-bar-wrap">0.978 <div class="metric-bar"><div class="metric-fill fill-green" style="width:97.8%"></div></div></div></td>
            <td>0.854</td>
            <td>0.882</td>
            <td>0.868</td>
          </tr>
          <tr>
            <td>◈ Random Forest</td>
            <td><div class="metric-bar-wrap">0.841 <div class="metric-bar"><div class="metric-fill fill-blue" style="width:84.1%"></div></div></div></td>
            <td><div class="metric-bar-wrap">0.971 <div class="metric-bar"><div class="metric-fill fill-blue" style="width:97.1%"></div></div></div></td>
            <td>0.826</td>
            <td>0.863</td>
            <td>0.844</td>
          </tr>
          <tr>
            <td>⬟ Decision Tree</td>
            <td><div class="metric-bar-wrap">0.631 <div class="metric-bar"><div class="metric-fill fill-amber" style="width:63.1%"></div></div></div></td>
            <td><div class="metric-bar-wrap">0.918 <div class="metric-bar"><div class="metric-fill fill-amber" style="width:91.8%"></div></div></div></td>
            <td>0.784</td>
            <td>0.607</td>
            <td>0.684</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div style="margin-top:20px; padding:16px 20px; background:rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.15); border-radius:8px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#9ca3af; reveal">
      <span style="color:var(--accent)">// WHY PR-AUC?</span> — A model classifying every transaction as legitimate achieves 99.83% accuracy while catching zero fraud.
      PR-AUC measures the precision-recall tradeoff on the minority class — the correct metric for heavily imbalanced datasets.
    </div>
  </div>
</section>

<!-- ═══════════════ IMPACT ═══════════════ -->
<section id="impact">
  <div class="container">
    <div class="section-label">Business Value</div>
    <h2 class="section-title reveal">Financial Impact</h2>
    <p class="section-desc reveal">Based on test set results at threshold 0.40 — $80 avg fraud loss, $5 per false positive review.</p>

    <div class="impact-grid reveal">
      <div class="impact-card green">
        <div class="impact-value">$39,200</div>
        <div class="impact-label">Fraud Caught</div>
      </div>
      <div class="impact-card red">
        <div class="impact-value">$6,800</div>
        <div class="impact-label">Fraud Missed</div>
      </div>
      <div class="impact-card blue">
        <div class="impact-value">$1,100</div>
        <div class="impact-label">Review Cost</div>
      </div>
      <div class="impact-card amber">
        <div class="impact-value">3,400%</div>
        <div class="impact-label">Net ROI</div>
      </div>
    </div>

    <div style="margin-top:16px; padding:16px 20px; background:var(--surface); border:1px solid var(--border); border-radius:8px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#9ca3af;">
      <span style="color:var(--accent2)">// THRESHOLD RATIONALE</span> — Youden's J statistic calibrates threshold at 0.40 rather than default 0.50.
      A missed fraud costs ~$80. A false positive costs ~$5 in analyst review time.
      Given the cost asymmetry, a lower threshold maximises net benefit.
    </div>
  </div>
</section>

<!-- ═══════════════ ARCHITECTURE ═══════════════ -->
<section id="architecture">
  <div class="container">
    <div class="section-label">System Design</div>
    <h2 class="section-title reveal">Architecture</h2>
    <p class="section-desc reveal">Clean separation between offline training and online serving. Each of the 11 pipeline phases logs start and end — a failed run immediately shows the exact failure point.</p>

    <div class="arch-grid reveal">
      <div class="arch-col">
        <div class="arch-col-header">⬡ Offline Training</div>
        <pre>creditcard.csv
      |
  loader.py · preprocessing.py
      |
  feature_engineering.py
  resampling.py (SMOTE)
      |
  train_model.py
  tuning.py  (Optuna)
      |
  MLflow experiment log
  models/*.pkl
  models/drift_baseline.json
      |
  reports/figures/*.png</pre>
      </div>
      <div class="arch-col">
        <div class="arch-col-header">◈ Online Serving</div>
        <pre>HTTP client / payment network
          |
     POST /predict
          |
  api/main.py  (FastAPI)
          |
  inference/predictor.py
          |
  xgboost_model.pkl
  scaler.pkl
  feature_names.pkl
          |
  model_monitor.py
          |
  dashboard/app.py (Streamlit)</pre>
      </div>
    </div>

    <!-- Collapsible project layout -->
    <div style="margin-top:24px;">
      <details class="reveal">
        <summary>
          <span style="color:var(--accent)">⬡</span> Project File Layout
          <span class="arrow">▶</span>
        </summary>
        <div class="details-body">
          <pre>fraud-detection/
├── src/
│   ├── data/           Data loading, cleaning, scaling, stratified splits
│   ├── features/       Feature engineering + SMOTE resampling
│   ├── training/       MLflow logging, Optuna tuning, 11-phase pipeline
│   ├── inference/      FraudPredictor class + Pydantic schemas
│   ├── models/         Evaluation: ROC/PR curves, confusion matrix
│   └── monitoring/     Drift detection (PSI + KS) + rolling health metrics
├── api/                FastAPI application — prediction endpoints
├── dashboard/          Streamlit analytics interface (5 pages)
├── monitoring/         Fraud alert dispatcher + JSONL event log
├── simulation/         Synthetic transaction stream for load testing
├── scripts/
│   ├── retrain.py      Drift-gated automated retraining + model promotion
│   └── evaluate.py     Standalone evaluation on any model + any CSV
├── tests/              Unit + integration tests (pytest, 39 tests)
├── notebooks/          EDA + model comparison (Jupytext percent-format)
├── docs/               Architecture, pipeline, API reference, system design
├── config/config.yaml  Single config file for all components
├── Makefile            All common operations as make targets
├── Dockerfile          Multi-stage production image
└── docker-compose.yml  Full stack: API, Dashboard, MLflow, Simulator</pre>
        </div>
      </details>

      <details class="reveal">
        <summary>
          <span style="color:var(--accent3)">◈</span> Engineering Decisions
          <span class="arrow">▶</span>
        </summary>
        <div class="details-body">
          <pre><span style="color:var(--accent)">Scaler fitted on training set only</span>
  → Prevents training-serving skew — the most common silent failure in deployed ML.
  → Saved to disk and loaded at inference time.

<span style="color:var(--accent)">Feature column order saved alongside model</span>
  → scikit-learn models are sensitive to column order.
  → Eliminates Python dict iteration order inconsistencies across environments.

<span style="color:var(--accent)">SMOTE applied after train/test split</span>
  → Applying SMOTE before splitting leaks synthetic samples into test set.
  → Inflated metrics, wrong model — this is Phase 4 in the pipeline.

<span style="color:var(--accent)">Threshold at 0.40 via Youden's J</span>
  → Calibrated on validation set, never test set.
  → Cost asymmetry: $80 missed fraud vs $5 false positive review.

<span style="color:var(--accent)">Dual drift detection: PSI + KS</span>
  → PSI is industry standard in credit risk monitoring.
  → KS catches distribution shifts PSI misses when changes fall within a bin boundary.</pre>
        </div>
      </details>
    </div>

    <!-- Feature engineering pills -->
    <div style="margin-top:28px;">
      <div class="section-label" style="margin-bottom:16px;">Feature Engineering</div>
      <div class="feature-list reveal">
        <div class="feature-item">
          <div class="feature-icon">🕐</div>
          <div class="feature-text">
            <h4>hour_of_day · is_night</h4>
            <p>Fraud skews heavily toward off-hours in card-present fraud studies</p>
          </div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">📊</div>
          <div class="feature-text">
            <h4>log_amount · amount_z</h4>
            <p>Compresses extreme right tail; MAD z-score for outlier detection</p>
          </div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">✕</div>
          <div class="feature-text">
            <h4>V1_V4 · V12_V14 · V1_V17</h4>
            <p>Interaction products between highest-importance PCA features</p>
          </div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">|x|</div>
          <div class="feature-text">
            <h4>V14_abs</h4>
            <p>Absolute value of V14 — the single strongest fraud predictor in published analyses</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════ API ═══════════════ -->
<section id="api">
  <div class="container">
    <div class="section-label">Endpoints</div>
    <h2 class="section-title reveal">API Reference</h2>
    <p class="section-desc reveal">FastAPI inference API with full Swagger UI at <code style="background:var(--code-bg); padding:2px 8px; border-radius:4px; font-size:12px; color:var(--accent); border:1px solid var(--border);">localhost:8000/docs</code></p>

    <div class="api-table-wrap reveal">
      <div class="api-row head">
        <div class="api-cell">Method</div>
        <div class="api-cell">Endpoint</div>
        <div class="api-cell">Description</div>
      </div>
      <div class="api-row">
        <div class="api-cell"><span class="method get">GET</span></div>
        <div class="api-cell">/health</div>
        <div class="api-cell">Liveness check with live fraud rate and P99 latency</div>
      </div>
      <div class="api-row">
        <div class="api-cell"><span class="method get">GET</span></div>
        <div class="api-cell">/info</div>
        <div class="api-cell">Model name, threshold, feature count, training metadata</div>
      </div>
      <div class="api-row">
        <div class="api-cell"><span class="method get">GET</span></div>
        <div class="api-cell">/metrics</div>
        <div class="api-cell">Operational metrics in JSON or Prometheus text format</div>
      </div>
      <div class="api-row">
        <div class="api-cell"><span class="method post">POST</span></div>
        <div class="api-cell">/predict</div>
        <div class="api-cell">Score a single transaction. Add <code>?explain=true</code> for SHAP values</div>
      </div>
      <div class="api-row">
        <div class="api-cell"><span class="method post">POST</span></div>
        <div class="api-cell">/predict/batch</div>
        <div class="api-cell">Score a CSV upload — up to 10,000 rows</div>
      </div>
    </div>

    <!-- Risk tiers -->
    <div style="margin-top:32px;">
      <div class="section-label" style="margin-bottom:16px;">Risk Tiers</div>
      <div class="tier-grid reveal">
        <div class="tier-card low">
          <div class="tier-name">⬡ Low</div>
          <div class="tier-range">&lt; 15%</div>
          <div class="tier-action">✓ Allow transaction</div>
        </div>
        <div class="tier-card medium">
          <div class="tier-name">◈ Medium</div>
          <div class="tier-range">15% – 40%</div>
          <div class="tier-action">⚠ Soft review</div>
        </div>
        <div class="tier-card high">
          <div class="tier-name">⬟ High</div>
          <div class="tier-range">40% – 70%</div>
          <div class="tier-action">✗ Manual review</div>
        </div>
        <div class="tier-card critical">
          <div class="tier-name">✦ Critical</div>
          <div class="tier-range">&gt; 70%</div>
          <div class="tier-action">⛔ Auto-block</div>
        </div>
      </div>
    </div>

    <!-- Example curl -->
    <div class="code-block reveal" style="margin-top:28px;">
      <div class="code-header">
        <span class="code-lang">bash — example request</span>
        <div class="code-dots">
          <div class="dot dot-red"></div>
          <div class="dot dot-amber"></div>
          <div class="dot dot-green"></div>
        </div>
      </div>
      <pre><span class="fn">curl</span> -s -X <span class="kw">POST</span> http://localhost:<span class="num">8000</span>/predict \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5364, "V4": 1.3782,
    "Amount": 149.62, "Time": 406.0
  }'</span>

<span class="cm"># Response:</span>
{
  <span class="str">"prediction"</span>: <span class="str">"legitimate"</span>,
  <span class="str">"probability"</span>: <span class="num">0.032</span>,
  <span class="str">"risk_tier"</span>: <span class="str">"LOW"</span>,
  <span class="str">"threshold_used"</span>: <span class="num">0.40</span>,
  <span class="str">"message"</span>: <span class="str">"Transaction appears normal. No action required."</span>
}</pre>
    </div>
  </div>
</section>

<!-- ═══════════════ SETUP ═══════════════ -->
<section id="setup">
  <div class="container">
    <div class="section-label">Getting Started</div>
    <h2 class="section-title reveal">Setup &amp; Commands</h2>

    <div class="code-block reveal">
      <div class="code-header">
        <span class="code-lang">bash — install</span>
        <div class="code-dots">
          <div class="dot dot-red"></div><div class="dot dot-amber"></div><div class="dot dot-green"></div>
        </div>
      </div>
      <pre><span class="fn">git</span> clone https://github.com/RagannagariSiva/fraud-detection.git
<span class="fn">cd</span> fraud-detection

python3 -m <span class="fn">venv</span> venv
<span class="fn">source</span> venv/bin/activate

<span class="fn">pip</span> install --upgrade pip
<span class="fn">pip</span> install -r requirements.txt</pre>
    </div>

    <div class="code-block reveal">
      <div class="code-header">
        <span class="code-lang">bash — run</span>
        <div class="code-dots">
          <div class="dot dot-red"></div><div class="dot dot-amber"></div><div class="dot dot-green"></div>
        </div>
      </div>
      <pre><span class="cm"># Train all 11 pipeline phases (~2 min on CPU)</span>
<span class="fn">python</span> main.py

<span class="cm"># Start inference API → http://localhost:8000/docs</span>
<span class="fn">uvicorn</span> api.main:app --host <span class="num">0.0.0.0</span> --port <span class="num">8000</span>

<span class="cm"># Start analytics dashboard → http://localhost:8501</span>
<span class="fn">streamlit</span> run dashboard/app.py

<span class="cm"># Run transaction simulator (2 TPS, 5% fraud rate)</span>
<span class="fn">python</span> simulation/real_time_transactions.py --tps <span class="num">2</span> --fraud-rate <span class="num">0.05</span>

<span class="cm"># Run all 39 tests</span>
<span class="fn">pytest</span> tests/ -v

<span class="cm"># Docker — full stack</span>
<span class="fn">docker</span> compose run --rm train
<span class="fn">docker</span> compose up api dashboard mlflow</pre>
    </div>

    <div class="code-block reveal">
      <div class="code-header">
        <span class="code-lang">makefile — shortcuts</span>
        <div class="code-dots">
          <div class="dot dot-red"></div><div class="dot dot-amber"></div><div class="dot dot-green"></div>
        </div>
      </div>
      <pre>make <span class="fn">train</span>       <span class="cm"># train full pipeline</span>
make <span class="fn">api</span>         <span class="cm"># start inference API</span>
make <span class="fn">dashboard</span>   <span class="cm"># start Streamlit dashboard</span>
make <span class="fn">simulate</span>    <span class="cm"># run transaction simulator</span>
make <span class="fn">test</span>        <span class="cm"># run test suite with coverage</span>
make <span class="fn">retrain</span>     <span class="cm"># drift-gated automated retraining</span>
make <span class="fn">docker-up</span>   <span class="cm"># start all services in Docker</span>
make <span class="fn">help</span>        <span class="cm"># full command reference</span></pre>
    </div>
  </div>
</section>

<!-- ═══════════════ TECH STACK ═══════════════ -->
<section style="border-bottom:none;">
  <div class="container">
    <div class="section-label">Stack</div>
    <h2 class="section-title reveal">Tech Stack</h2>
    <div class="tech-grid reveal">
      <div class="tech-pill">🐍 Python 3.10+</div>
      <div class="tech-pill">⬡ XGBoost</div>
      <div class="tech-pill">◈ scikit-learn</div>
      <div class="tech-pill">⚡ FastAPI</div>
      <div class="tech-pill">📊 Streamlit</div>
      <div class="tech-pill">✦ MLflow</div>
      <div class="tech-pill">🔬 Optuna</div>
      <div class="tech-pill">🧠 SHAP</div>
      <div class="tech-pill">⚖ imbalanced-learn</div>
      <div class="tech-pill">📐 Pydantic v2</div>
      <div class="tech-pill">📉 scipy</div>
      <div class="tech-pill">🧪 pytest</div>
      <div class="tech-pill">🐳 Docker</div>
      <div class="tech-pill">📦 ruff</div>
    </div>

    <div style="margin-top:32px; padding:20px; background:var(--surface); border:1px solid var(--border); border-radius:10px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; line-height:1.7;">
      <div style="color:var(--accent); margin-bottom:6px;">// DATASET</div>
      <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" style="color:#60a5fa; text-decoration:none;">Credit Card Fraud Detection</a> by ULB Machine Learning Group · Open Database License<br>
      284,807 transactions · September 2013 · European cardholders · 0.172% fraud rate (492 of 284,807)<br>
      V1–V28 are PCA-transformed components. Amount and Time are raw values. Cannot be redistributed.
    </div>
  </div>
</section>

<!-- ═══════════════ FOOTER ═══════════════ -->
<footer>
  <div class="container">
    <div class="logo">FraudGuard</div>
    <p>Built by <a href="https://github.com/RagannagariSiva" target="_blank">Ragannagari Siva</a> · <a href="https://github.com/RagannagariSiva/fraud-detection" target="_blank">GitHub Repository</a></p>
    <p style="margin-top:6px;">Production-grade ML system · End-to-end · XGBoost · FastAPI · Streamlit</p>
  </div>
</footer>

<script>
  // Cursor glow
  const glow = document.getElementById('cursorGlow');
  document.addEventListener('mousemove', e => {
    glow.style.left = e.clientX + 'px';
    glow.style.top = e.clientY + 'px';
  });

  // Reveal on scroll
  const reveals = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver(entries => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 60);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  reveals.forEach(el => observer.observe(el));

  // TOC dots active state
  const sections = ['hero','dashboard','performance','impact','architecture','api','setup'];
  const tocDots = document.querySelectorAll('.toc-dot');

  const secObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        tocDots.forEach(d => d.classList.remove('active'));
        const active = document.querySelector(`.toc-dot[data-target="${id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { threshold: 0.4 });

  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) secObserver.observe(el);
  });

  function scrollTo(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  }
</script>
</body>
</html>
