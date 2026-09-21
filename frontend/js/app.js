// Shell: auth screen, navigation, theme-aware chart redraw.
let currentView = "overview";
let registering = false;

const $ = (sel) => document.querySelector(sel);

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

async function navigate(view) {
  currentView = VIEWS[view] ? view : "overview";
  destroyChart();
  document.querySelectorAll("#nav button").forEach((b) => {
    if (b.dataset.view === currentView) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  history.replaceState(null, "", "#" + currentView);
  const root = $("#view");
  root.innerHTML = "";
  await VIEWS[currentView](root);
}

function showAuth() {
  $("#app").classList.add("hidden");
  $("#auth").classList.remove("hidden");
}

async function showApp() {
  $("#auth").classList.add("hidden");
  $("#app").classList.remove("hidden");
  try { $("#user").textContent = (await api("/auth/me")).name; } catch { return; }
  navigate(location.hash.slice(1) || "overview");
}

function setMode(reg) {
  registering = reg;
  $("#name-field").classList.toggle("hidden", !reg);
  $("#name-field input").required = reg;
  $("#auth-submit").textContent = reg ? "Create account" : "Sign in";
  $("#auth-toggle").textContent = reg ? "I already have an account" : "Create an account";
  $("#auth-error").textContent = "";
}

$("#auth-toggle").onclick = () => setMode(!registering);
$("#auth-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const f = Object.fromEntries(new FormData(ev.target));
  $("#auth-error").textContent = "";
  try {
    if (registering) await api("/auth/register", { method: "POST", body: { name: f.name, email: f.email, password: f.password } });
    const tok = await api("/auth/login", { method: "POST", body: { email: f.email, password: f.password } });
    store.set(tok.access_token);
    ev.target.reset();
    showApp();
  } catch (e) { $("#auth-error").textContent = e.message; }
};

$("#nav").onclick = (ev) => { const b = ev.target.closest("button[data-view]"); if (b) navigate(b.dataset.view); };
$("#logout").onclick = () => { store.clear(); showAuth(); };
window.addEventListener("auth-expired", showAuth);
// Chart colours are read from CSS variables, so redraw when the OS theme flips.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (currentView === "budgets") navigate("budgets"); });

store.get() ? showApp() : (setMode(false), showAuth());
