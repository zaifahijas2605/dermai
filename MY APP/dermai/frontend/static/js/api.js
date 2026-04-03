
const API_BASE = "/api/v1";


const TokenStore = {
  setTokens(access, refresh) {
    sessionStorage.setItem("access_token", access);
    sessionStorage.setItem("refresh_token", refresh);
  },
  getAccess()  { return sessionStorage.getItem("access_token"); },
  getRefresh() { return sessionStorage.getItem("refresh_token"); },
  clear() {
    sessionStorage.removeItem("access_token");
    sessionStorage.removeItem("refresh_token");
  },
  isLoggedIn() { return !!this.getAccess(); },
};


async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = TokenStore.getAccess();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

 
  if (res.status === 401 && TokenStore.getRefresh()) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      headers["Authorization"] = `Bearer ${TokenStore.getAccess()}`;
      return fetch(`${API_BASE}${path}`, { ...options, headers });
    } else {
      TokenStore.clear();
      window.location.href = "/login.html";
      return res;
    }
  }
  return res;
}

async function tryRefresh() {
  const rt = TokenStore.getRefresh();
  if (!rt) return false;
  const res = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: rt }),
  });
  if (!res.ok) return false;
  const data = await res.json();
  TokenStore.setTokens(data.access_token, data.refresh_token);
  return true;
}


const AuthAPI = {
  async register(email, displayName, password) {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, display_name: displayName, password }),
    });
    return { ok: res.ok, data: await res.json(), status: res.status };
  },

  async login(email, password) {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (res.ok) TokenStore.setTokens(data.access_token, data.refresh_token);
    return { ok: res.ok, data, status: res.status };
  },

  async logout() {
    const rt = TokenStore.getRefresh();
    if (rt) {
      await apiFetch("/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: rt }),
      }).catch(() => {});
    }
    TokenStore.clear();
    window.location.href = "/login.html";
  },

  async getMe() {
    const res = await apiFetch("/auth/me");
    return { ok: res.ok, data: await res.json() };
  },
};


const PredictAPI = {
  async predict(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await apiFetch("/predict", { method: "POST", body: form });
    return { ok: res.ok, data: await res.json(), status: res.status };
  },
};


function requireAuth() {
  if (!TokenStore.isLoggedIn()) {
    window.location.href = "/login.html";
  }
}


function extractError(data) {
  if (!data) return "An unexpected error occurred.";
  if (data.detail) {
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map(e => e.msg || String(e)).join(" ");
    }
  }
  return "An unexpected error occurred.";
}
