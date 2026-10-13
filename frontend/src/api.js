const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export const getAnalytics = () => request("/analytics/summary");
export const getTickets = () => request("/tickets");

export const createTicket = (message) =>
  request("/tickets", {
    method: "POST",
    body: JSON.stringify({ message }),
  });

export const updateTicketStatus = (id, status) =>
  request(`/tickets/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });

export const analyzeTicket = (message) =>
  request("/predict", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
