// Thin API client. The token lives in localStorage (wrapped: storage can be unavailable).
const API_BASE = window.API_BASE || "http://localhost:8000";

const store = {
  get() { try { return localStorage.getItem("token"); } catch { return null; } },
  set(v) { try { localStorage.setItem("token", v); } catch { /* ignore */ } },
  clear() { try { localStorage.removeItem("token"); } catch { /* ignore */ } },
};

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, { method = "GET", body, form } = {}) {
  const headers = {};
  const token = store.get();
  if (token) headers.Authorization = `Bearer ${token}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }

  let res;
  try {
    res = await fetch(API_BASE + path, { method, headers, body: payload });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}. Is the backend running?`);
  }
  if (res.status === 401 && token) { store.clear(); window.dispatchEvent(new Event("auth-expired")); }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((e) => e.msg).join("; ");
    throw new ApiError(res.status, msg || `Request failed (${res.status})`);
  }
  return data;
}
