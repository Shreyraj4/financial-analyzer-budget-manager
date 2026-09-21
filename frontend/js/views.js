// Views. Everything that comes from the server (merchant names especially, which are
// attacker-controllable text in a bank statement) goes through esc() before touching innerHTML.
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const inr = (n) => "₹" + Math.abs(Number(n)).toLocaleString("en-IN", { maximumFractionDigits: 0 }); // amounts are stored signed; type says direction
const fmtDate = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
const pct = (x) => (x * 100).toFixed(0) + "%";
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SEV_ICON = { info: "ℹ", positive: "✓", watch: "!", alert: "▲" };
const ST_ICON = { ok: "✓", warning: "!", exceeded: "▲" };

let activeChart = null;
function destroyChart() { if (activeChart) { activeChart.destroy(); activeChart = null; } }

function errorBox(e) { return `<div class="notice error">${esc(e.message)}</div>`; }

/* ------------------------------------------------------------------ Overview */
async function viewOverview(root) {
  root.innerHTML = `<div class="card"><div class="row"><h2 style="margin:0;flex:1">Your spending report</h2>
      <span id="narr" class="muted small"></span>
      <button class="btn" id="gen">Generate report</button></div>
      <div id="report" class="muted" style="margin-top:12px">Loading…</div></div>
    <div id="profile"></div>`;

  const renderReport = (r) => {
    const box = root.querySelector("#report");
    if (!r) { box.innerHTML = "No report yet. Import transactions, then click <b>Generate report</b>."; return; }
    const facts = Object.fromEntries(r.facts.map((f) => [f.id, f.statement]));
    const c = r.content;
    const src = (ids) => ids.length
      ? `<details class="sources"><summary>Sources (${ids.length})</summary><ul>${ids.map((i) => `<li>${esc(facts[i] || i)}</li>`).join("")}</ul></details>` : "";
    const v = r.verification, n = r.narrator;
    const note = v.fell_back ? `Written by the built-in template (the AI writer failed: ${esc(v.error || "unknown error")}).`
      : n.used === "template" ? "Written by the built-in template (no AI key set)."
      : `Written by ${esc(n.model)}.`;
    box.className = "";
    box.innerHTML = `
      <h3>${esc(c.headline)}</h3><p>${esc(c.summary)}</p>
      ${c.insights.map((i) => `<div class="insight sev-${esc(i.severity)}">
          <h3><span class="badge sev-${esc(i.severity)}" data-icon="${SEV_ICON[i.severity] || ""}">${esc(i.severity)}</span>${esc(i.title)}</h3>
          <p>${esc(i.text)}</p>${src(i.fact_ids)}</div>`).join("")}
      ${c.suggested_actions.length ? `<h3>Suggested actions</h3><ul>${c.suggested_actions.map((a) => `<li>${esc(a.text)}${src(a.fact_ids)}</li>`).join("")}</ul>` : ""}
      <p class="muted small">${note} ${v.numbers_checked} numbers checked against the facts; ${(v.dropped_parts || []).length} unverifiable part(s) removed.
      Generated ${esc(new Date(r.created_at).toLocaleString())}.</p>`;
  };

  try {
    const [st, latest] = await Promise.all([api("/reports/status"), api("/reports/latest")]);
    root.querySelector("#narr").textContent = st.narrator === "template" ? "Writer: template" : `Writer: ${st.model}`;
    renderReport(latest);
  } catch (e) { root.querySelector("#report").innerHTML = errorBox(e); }

  root.querySelector("#gen").onclick = async (ev) => {
    const btn = ev.target; btn.disabled = true; btn.textContent = "Generating…";
    try { renderReport(await api("/reports/generate", { method: "POST" })); toast("Report generated"); }
    catch (e) { toast(e.message); }
    btn.disabled = false; btn.textContent = "Generate report";
  };

  try {
    const p = await api("/analytics/spending-profile");
    const box = root.querySelector("#profile");
    if (!p) { box.innerHTML = `<div class="card"><h2>Spending profile</h2><p class="muted">Not enough history yet.</p></div>`; return; }
    const max = Math.max(...p.your_top_categories.map((c) => c.share), ...p.profile_top_categories.map((c) => c.share), 0.01);
    const bars = (list, cls) => list.map((c) => `<div class="bar-row"><span>${esc(c.category)}</span>
        <div class="bar-track"><div class="bar-fill ${cls}" style="width:${(c.share / max) * 100}%"></div></div><span class="num">${pct(c.share)}</span></div>`).join("");
    box.innerHTML = `<div class="card"><h2>Spending profile: ${esc(p.name)}</h2>
      <p class="muted">Based on your last ${p.window_months} months${p.is_full_window ? "" : " (less than a full window)"}.
      Typical monthly spend: <b>${inr(p.your_typical_monthly_spend)}</b> vs ${inr(p.profile_typical_monthly_spend)} for this profile.
      ${p.assignment_margin < 0.2 ? "You sit close to the border between two profiles." : ""}</p>
      <div class="grid"><div><h3>Your top categories</h3>${bars(p.your_top_categories, "you")}</div>
      <div><h3>Typical for this profile</h3>${bars(p.profile_top_categories, "profile")}</div></div></div>`;
  } catch (e) { root.querySelector("#profile").innerHTML = errorBox(e); }
}

/* ------------------------------------------------------------------ Budgets */
async function viewBudgets(root) {
  root.innerHTML = `<div class="card"><h2>Budget status</h2><div id="status" class="muted">Loading…</div></div>
    <div class="card"><h2>Add a budget</h2><form id="bform" class="form-grid">
      <label>Category<input name="category" required></label>
      <label>Amount (₹)<input name="amount" type="number" min="1" step="0.01" required></label>
      <label>Period<select name="period"><option>monthly</option><option>weekly</option></select></label>
      <label>Start<input name="start_date" type="date" required></label>
      <label>End<input name="end_date" type="date" required></label>
      <div><button class="btn" type="submit">Add budget</button></div></form><p id="berr" class="error small"></p></div>
    <div class="card"><h2>Recommended budgets for next month</h2><div id="recs" class="muted">Loading…</div></div>`;

  const loadStatus = async () => {
    const box = root.querySelector("#status");
    try {
      const list = await api("/budgets/status");
      if (!list.length) { box.textContent = "No budgets yet. Add one below, or use the recommendations."; return; }
      box.className = "";
      box.innerHTML = list.map((s) => {
        const used = Number(s.percent_used);
        return `<div class="bar-row" style="grid-template-columns:9rem 1fr 15rem">
          <span>${esc(s.category)}<br><span class="muted small">${esc(s.period)}, ${fmtDate(s.start_date)} to ${fmtDate(s.end_date)}</span></span>
          <div class="bar-track"><div class="bar-fill ${esc(s.status)}" style="width:${Math.min(used, 100)}%"></div></div>
          <span>${inr(s.spent)} of ${inr(s.budget_amount)}
            <span class="badge st-${esc(s.status)}" data-icon="${ST_ICON[s.status] || ""}">${esc(s.status)} ${used.toFixed(0)}%</span>
            <button class="link small" data-del="${s.budget_id}" aria-label="Delete budget">delete</button></span></div>`;
      }).join("");
      box.querySelectorAll("[data-del]").forEach((b) => b.onclick = async () => {
        try { await api(`/budgets/${b.dataset.del}`, { method: "DELETE" }); loadStatus(); } catch (e) { toast(e.message); }
      });
    } catch (e) { box.innerHTML = errorBox(e); }
  };
  loadStatus();

  root.querySelector("#bform").onsubmit = async (ev) => {
    ev.preventDefault();
    const f = Object.fromEntries(new FormData(ev.target));
    root.querySelector("#berr").textContent = "";
    try {
      await api("/budgets", { method: "POST", body: { ...f, amount: Number(f.amount) } });
      ev.target.reset(); loadStatus(); toast("Budget added");
    } catch (e) { root.querySelector("#berr").textContent = e.message; }
  };

  try {
    const r = await api("/recommendations/budgets");
    const box = root.querySelector("#recs");
    if (!r) { box.textContent = "Not enough history to recommend budgets yet."; return; }
    box.className = "";
    const recs = r.recommendations;
    const ACTION = { set_budget: "Set this budget", raise: "Raise budget", tighten: "Tighten budget", keep: "Keep as is" };
    box.innerHTML = `<p class="muted">For ${new Date(r.for_month + "T00:00:00").toLocaleDateString("en-IN", { month: "long", year: "numeric" })}.
      Each budget is the ML forecast plus a margin calibrated so you stay within it about ${pct(r.target_coverage)} of months.
      Total forecast <b>${inr(r.total_forecast)}</b>, total recommended <b>${inr(r.total_recommended)}</b>.
      ${r.excluded_anomalous_transactions ? `Excluded ${r.excluded_anomalous_transactions} unusual transaction(s) (${inr(r.excluded_anomalous_spend)}) from the history.` : ""}</p>
      <div class="legend"><span><i style="background:var(--series-1)"></i>Forecast</span><span><i style="background:var(--series-2)"></i>Recommended budget</span></div>
      <div class="chart-box"><canvas id="rchart" role="img" aria-label="Forecast and recommended budget per category"></canvas></div>
      <details style="margin-top:12px"><summary>Show as table</summary><div class="table-wrap"><table><thead><tr>
        <th>Category</th><th class="num">Forecast</th><th class="num">Budget</th><th class="num">Last month</th><th class="num">vs 3-mo avg</th><th>Suggestion</th></tr></thead><tbody>
      ${recs.map((x) => `<tr><td>${esc(x.category)}</td><td class="num">${inr(x.forecast)}</td><td class="num">${inr(x.recommended_budget)}</td>
        <td class="num">${inr(x.last_month_spend)}</td><td class="num">${x.forecast_vs_avg_3_month_pct == null ? "–" : x.forecast_vs_avg_3_month_pct.toFixed(1) + "%"}</td>
        <td>${ACTION[x.action] || esc(x.action)}${x.existing_budget != null ? ` (current ${inr(x.existing_budget)})` : ""}</td></tr>`).join("")}
      </tbody></table></div></details>`;
    drawRecChart(recs, root.querySelector("#rchart"));
  } catch (e) { root.querySelector("#recs").innerHTML = errorBox(e); }
}

function drawRecChart(recs, canvas) {
  destroyChart();
  if (typeof Chart === "undefined") return;
  const ink = cssVar("--ink-2"), grid = cssVar("--grid"), surface = cssVar("--surface");
  const sorted = [...recs].sort((a, b) => b.recommended_budget - a.recommended_budget);
  activeChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: sorted.map((r) => r.category),
      datasets: [
        { label: "Forecast", data: sorted.map((r) => r.forecast), backgroundColor: cssVar("--series-1"), borderColor: surface, borderWidth: 2, borderRadius: 4 },
        { label: "Recommended budget", data: sorted.map((r) => r.recommended_budget), backgroundColor: cssVar("--series-2"), borderColor: surface, borderWidth: 2, borderRadius: 4 },
      ],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${inr(c.parsed.x)}` } } },
      scales: {
        x: { beginAtZero: true, grid: { color: grid }, ticks: { color: ink, callback: (v) => inr(v) } },
        y: { grid: { display: false }, ticks: { color: ink } },
      },
    },
  });
}

/* ------------------------------------------------------------------ Transactions */
async function viewTransactions(root) {
  root.innerHTML = `<div class="card" id="review-card"><h2>Needs your label</h2><div id="review" class="muted">Loading…</div></div>
    <div class="card"><h2>Recent transactions</h2><div id="list" class="muted">Loading…</div>
      <p><button class="btn secondary hidden" id="more">Load more</button></p></div>`;
  let categories = [];
  try { categories = await api("/transactions/categories"); } catch { /* labeling just gets a free-text box */ }
  const catInput = (id) => `<input list="cats" data-cat="${id}" placeholder="Category" style="width:11rem">`;
  root.insertAdjacentHTML("beforeend", `<datalist id="cats">${categories.map((c) => `<option value="${esc(c)}">`).join("")}</datalist>`);

  const txRows = (list, withLabel) => list.map((t) => `<tr><td>${fmtDate(t.transaction_date)}</td><td>${esc(t.description)}</td>
      <td class="num">${t.transaction_type === "credit" ? "+" : ""}${inr(t.amount)}</td>
      <td>${withLabel ? catInput(t.id) + ` <button class="btn" data-label="${t.id}">Save</button>` : esc(t.category || "–")}</td></tr>`).join("");
  const table = (rows, last) => `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Description</th><th class="num">Amount</th><th>${last}</th></tr></thead><tbody>${rows}</tbody></table></div>`;

  const loadReview = async () => {
    const box = root.querySelector("#review");
    try {
      const list = await api("/transactions/review?limit=25");
      if (!list.length) { box.textContent = "Nothing to review: every transaction has a category."; return; }
      box.className = "";
      box.innerHTML = `<p class="muted small">The model wasn't confident about these. Your label is also applied to other transactions from the same merchant and remembered for future imports.</p>` + table(txRows(list, true), "Category");
      box.querySelectorAll("[data-label]").forEach((b) => b.onclick = async () => {
        const val = box.querySelector(`[data-cat="${b.dataset.label}"]`).value.trim();
        if (!val) return;
        try {
          const r = await api(`/transactions/${b.dataset.label}/category`, { method: "PATCH", body: { category: val, apply_to_merchant: true } });
          toast(`Labeled ${r.updated_count} transaction(s) as ${val}`); loadReview(); reloadList();
        } catch (e) { toast(e.message); }
      });
    } catch (e) { box.innerHTML = errorBox(e); }
  };

  let offset = 0; const PAGE = 50;
  const listBox = root.querySelector("#list"), more = root.querySelector("#more");
  const loadPage = async (reset) => {
    if (reset) { offset = 0; listBox.innerHTML = ""; }
    try {
      const list = await api(`/transactions?limit=${PAGE}&offset=${offset}`);
      if (reset && !list.length) { listBox.textContent = "No transactions yet. Use the Upload tab."; more.classList.add("hidden"); return; }
      listBox.className = "";
      if (reset) listBox.innerHTML = table("", "Category");
      listBox.querySelector("tbody").insertAdjacentHTML("beforeend", txRows(list, false));
      offset += list.length;
      more.classList.toggle("hidden", list.length < PAGE);
    } catch (e) { listBox.innerHTML = errorBox(e); }
  };
  const reloadList = () => loadPage(true);
  more.onclick = () => loadPage(false);
  loadReview(); loadPage(true);
}

/* ------------------------------------------------------------------ Upload */
async function viewUpload(root) {
  root.innerHTML = `<div class="card"><h2>Upload a bank statement</h2>
      <p class="muted small">CSV or PDF, up to 5 MB. You'll see a preview and can fix categories before anything is saved.</p>
      <div class="row"><input type="file" id="file" accept=".csv,.pdf"><button class="btn" id="up">Preview</button></div>
      <p id="uerr" class="error small"></p></div><div id="preview"></div>`;
  let categories = [];
  try { categories = await api("/transactions/categories"); } catch { /* ignore */ }

  root.querySelector("#up").onclick = async () => {
    const file = root.querySelector("#file").files[0];
    const err = root.querySelector("#uerr"); err.textContent = "";
    if (!file) { err.textContent = "Choose a file first."; return; }
    const form = new FormData(); form.append("file", file);
    const btn = root.querySelector("#up"); btn.disabled = true;
    try { renderPreview(root.querySelector("#preview"), await api("/transactions/upload", { method: "POST", form }), categories); }
    catch (e) { err.textContent = e.message; }
    btn.disabled = false;
  };
}

function renderPreview(box, p, categories) {
  const opts = (sel) => `<option value="">(auto)</option>` + categories.map((c) => `<option ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("");
  box.innerHTML = `<div class="card"><h2>${esc(p.filename)}</h2>
    <p>${p.valid_count} valid row(s), ${p.error_count} with errors (skipped).
    ${p.rows.filter((r) => r.needs_review).length} row(s) the model is unsure about are highlighted: pick a category, or leave them to label later.</p>
    <div class="table-wrap"><table><thead><tr><th>Date</th><th>Description</th><th class="num">Amount</th><th>Suggested</th><th>Your label</th></tr></thead><tbody>
    ${p.rows.map((r, i) => `<tr ${r.valid ? "" : 'style="opacity:.55"'}>
      <td>${r.transaction_date ? fmtDate(r.transaction_date) : "–"}</td><td>${esc(r.description)}</td>
      <td class="num">${r.amount == null ? "–" : inr(r.amount)}</td>
      <td>${r.valid ? (r.category ? `${esc(r.category)} <span class="muted small">${r.confidence != null ? pct(r.confidence) : esc(r.category_source || "")}</span>`
          : `<span class="badge sev-watch" data-icon="?">unsure${r.suggested_category ? ": " + esc(r.suggested_category) : ""}</span>`) : `<span class="error small">${esc(r.errors.join("; "))}</span>`}</td>
      <td>${r.valid ? `<select data-i="${i}">${opts("")}</select>` : ""}</td></tr>`).join("")}
    </tbody></table></div>
    <p><button class="btn" id="imp" ${p.valid_count ? "" : "disabled"}>Import ${p.valid_count} transaction(s)</button></p></div>`;

  box.querySelector("#imp").onclick = async (ev) => {
    const payload = [];
    p.rows.forEach((r, i) => {
      if (!r.valid) return;
      const chosen = box.querySelector(`select[data-i="${i}"]`).value;
      payload.push({
        transaction_date: r.transaction_date, description: r.description, merchant: r.merchant,
        amount: r.amount, transaction_type: r.transaction_type,
        ...(chosen ? { category: chosen } : {}),
      });
    });
    ev.target.disabled = true;
    try {
      const res = await api("/transactions/import", { method: "POST", body: { transactions: payload } });
      box.innerHTML = `<div class="card"><h2>Import complete</h2><p>${res.imported_count} imported, ${res.duplicate_count} duplicate(s) skipped.</p>
        <button class="btn" onclick="navigate('transactions')">View transactions</button></div>`;
    } catch (e) { toast(e.message); ev.target.disabled = false; }
  };
}

/* ------------------------------------------------------------------ Anomalies */
async function viewAnomalies(root) {
  root.innerHTML = `<div class="card"><h2>Unusual transactions</h2>
      <p class="muted small">Flagged by an Isolation Forest model. Each reason is computed in code, not written by an LLM. Some flags will be false alarms.</p>
      <div id="txa" class="muted">Loading…</div></div>
    <div class="card"><h2>Unusual category totals</h2>
      <p class="muted small">A simple statistical baseline: months where a category's total was far above its usual level (z-score).</p>
      <div id="cata" class="muted">Loading…</div></div>`;
  try {
    const list = await api("/analytics/transaction-anomalies");
    const box = root.querySelector("#txa");
    if (!list.length) { box.textContent = "Nothing unusual found."; }
    else {
      box.className = "";
      box.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Description</th><th class="num">Amount</th><th>Why it was flagged</th></tr></thead><tbody>
        ${list.map((a) => `<tr><td>${fmtDate(a.transaction_date)}</td><td>${esc(a.description)}<br><span class="muted small">${esc(a.category || "")}</span></td>
          <td class="num">${inr(a.amount)}</td><td>${a.reasons.map((r) => esc(r.text)).join("<br>")}</td></tr>`).join("")}</tbody></table></div>`;
    }
  } catch (e) { root.querySelector("#txa").innerHTML = errorBox(e); }
  try {
    const list = await api("/analytics/anomalies");
    const box = root.querySelector("#cata");
    if (!list.length) { box.textContent = "No unusual category months."; return; }
    box.className = "";
    box.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Category</th><th>Month</th><th class="num">Spent</th><th class="num">Usual</th><th class="num">z-score</th></tr></thead><tbody>
      ${list.map((a) => `<tr><td>${esc(a.category)}</td><td>${a.year}-${String(a.month).padStart(2, "0")}</td><td class="num">${inr(a.total_spent)}</td>
        <td class="num">${inr(a.baseline_mean)}</td><td class="num">${a.z_score.toFixed(1)}</td></tr>`).join("")}</tbody></table></div>`;
  } catch (e) { root.querySelector("#cata").innerHTML = errorBox(e); }
}

const VIEWS = { overview: viewOverview, budgets: viewBudgets, transactions: viewTransactions, upload: viewUpload, anomalies: viewAnomalies };
