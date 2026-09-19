// Vite proxies /api during development; the production image serves the API
// and the built SPA from the same origin, so it uses root-relative endpoints.
const API_BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "/api" : "");

export async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch { /* Keep the HTTP error. */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

export const getDocuments = () => request("/documents");
export const deleteDocument = id => request(`/documents/${id}`, { method: "DELETE" });
export const uploadDocument = file => {
  const body = new FormData();
  body.append("file", file);
  return request("/documents/upload", { method: "POST", body });
};
export const search = payload => request("/search", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
});
export const ask = payload => request("/questions", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
});
export const getEvaluations = () => request("/evaluations?limit=40");
export const compareEvaluations = () => request("/evaluations/compare", { method: "POST" });
