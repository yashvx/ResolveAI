import { useCallback, useEffect, useState } from "react";
import {
  Activity, AlertTriangle, ArrowLeft, BrainCircuit,
  CheckCircle2, Clock3, RefreshCw, Search, Ticket,
} from "lucide-react";
import {
  analyzeTicket, createTicket, getAnalytics, getTickets, updateTicketStatus,
} from "./api";
import "./Dashboard.css";

const EMPTY = {
  total_tickets: 0,
  open_tickets: 0,
  in_progress_tickets: 0,
  resolved_tickets: 0,
  high_priority_tickets: 0,
  needs_review_tickets: 0,
  categories: {},
};

export default function Dashboard({ onBack }) {
  const [analytics, setAnalytics] = useState(EMPTY);
  const [tickets, setTickets] = useState([]);
  const [message, setMessage] = useState("");
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [summary, records] = await Promise.all([
        getAnalytics(),
        getTickets(),
      ]);
      setAnalytics(summary);
      setTickets(records);
    } catch (err) {
      setError(`${err.message}. Is the FastAPI server running?`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleAnalyze() {
    if (message.trim().length < 5) return;
    setAnalyzing(true);
    setError("");
    setNotice("");
    try {
      const result = await analyzeTicket(message.trim());
      setAnalysis(result);
    } catch (err) {
      setError(err.message);
      setAnalysis(null);
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleCreate(event) {
    event.preventDefault();
    if (message.trim().length < 5) return;
    setSaving(true);
    setError("");
    setNotice("");

    try {
      const ticket = await createTicket(message.trim());
      setMessage("");
      setNotice(
        `Ticket #${ticket.id} created: ${ticket.category}, ${ticket.priority} priority.`
      );
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleStatus(ticket, status) {
    setError("");
    setNotice("");
    try {
      await updateTicketStatus(ticket.id, status);
      setNotice(`Ticket #${ticket.id} updated to ${status}.`);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  const filtered = tickets.filter((ticket) => {
    const term = search.toLowerCase();
    const matchesText =
      ticket.message.toLowerCase().includes(term) ||
      ticket.category.toLowerCase().includes(term) ||
      String(ticket.id).includes(term);
    return matchesText &&
      (statusFilter === "All" || ticket.status === statusFilter);
  });

  const stats = [
    { label: "Total tickets", value: analytics.total_tickets, icon: Ticket },
    { label: "Open tickets", value: analytics.open_tickets, icon: Clock3 },
    { label: "High priority", value: analytics.high_priority_tickets, icon: AlertTriangle },
    { label: "Resolved", value: analytics.resolved_tickets, icon: CheckCircle2 },
  ];

  return (
    <div className="workspace">
      <aside className="workspace-sidebar">
        <a className="workspace-brand" href="#dashboard">
          <span className="workspace-logo"><BrainCircuit size={22} /></span>
          <span>Resolve<span>AI</span></span>
        </a>
        <p className="sidebar-label">WORKSPACE</p>
        <div className="sidebar-link selected"><Activity size={18} /> Overview</div>
        <div className="sidebar-link"><Ticket size={18} /> All tickets</div>
        <div className="sidebar-link"><BrainCircuit size={18} /> AI analysis</div>
        <div className="sidebar-bottom"><span className="online-dot" /> Local API workspace</div>
      </aside>

      <main className="workspace-main">
        <header className="workspace-header">
          <div>
            <p className="workspace-eyebrow">WORKSPACE / OVERVIEW</p>
            <h1>Support overview</h1>
            <p className="muted">Monitor and manage your customer support tickets.</p>
          </div>
          <button className="secondary-button" onClick={onBack}>
            <ArrowLeft size={16} /> Landing page
          </button>
        </header>

        {error && <div className="notice error">{error}</div>}
        {notice && <div className="notice success">{notice}</div>}

        <section className="stats-grid">
          {stats.map(({ label, value, icon: Icon }) => (
            <article className="stat-card" key={label}>
              <div className="stat-label">{label}<Icon size={19} /></div>
              <strong>{value}</strong>
              <span>Live database metric</span>
            </article>
          ))}
        </section>

        <section className="dashboard-grid">
          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="panel-eyebrow">AI WORKFLOW</span>
                <h2>Analyze a new ticket</h2>
              </div>
              <BrainCircuit size={21} />
            </div>
            <p className="muted">
              Classify a customer message and save the prediction to your database.
            </p>
            <form onSubmit={handleCreate}>
              <label htmlFor="ticket-message">Customer message</label>
              <textarea
                id="ticket-message"
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="My payment was deducted twice, but my order failed..."
                minLength={5}
                maxLength={3000}
                required
              />
              <div className="form-footer">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleAnalyze}
                  disabled={analyzing || message.trim().length < 5}
                >
                  {analyzing ? "Analyzing..." : "Analyze only"}
                </button>
                <span>{message.length}/3000 characters</span>
                <button className="primary-button" disabled={saving || message.trim().length < 5}>
                  {saving ? "Analyzing..." : "Analyze & save"}
                </button>
              </div>
            </form>
            {analysis && (
              <div className={`analysis-result ${analysis.needs_review ? "analysis-warning" : "analysis-ok"}`}>
                <div className="analysis-result-heading">
                  <BrainCircuit size={19} />
                  <strong>AI analysis result</strong>
                </div>
                <div className="analysis-metrics">
                  <div>
                    <span>Predicted category</span>
                    <strong>{analysis.category}</strong>
                  </div>
                  <div>
                    <span>Model confidence</span>
                    <strong>{(analysis.confidence * 100).toFixed(2)}%</strong>
                  </div>
                </div>
                <p>
                  {analysis.needs_review
                    ? "Human review recommended because the model confidence is low."
                    : "The model confidence meets the current review threshold."}
                </p>
              </div>
            )}
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="panel-eyebrow">CLASSIFICATION</span>
                <h2>Ticket categories</h2>
              </div>
              <Activity size={21} />
            </div>
            <div className="category-list">
              {Object.entries(analytics.categories || {}).length === 0
                ? <p className="muted">Categories appear when tickets are saved.</p>
                : Object.entries(analytics.categories).map(([category, count]) => (
                  <div key={category} className="category-item">
                    <div className="category-label"><span>{category}</span><strong>{count}</strong></div>
                    <div className="category-track">
                      <div
                        className="category-fill"
                        style={{
                          width: `${analytics.total_tickets ? count / analytics.total_tickets * 100 : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>
            <div className="review-summary">
              <AlertTriangle size={17} />
              <span>{analytics.needs_review_tickets} ticket(s) flagged for human review</span>
            </div>
          </article>
        </section>

        <section className="panel table-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">SUPPORT QUEUE</span>
              <h2>Recent tickets</h2>
            </div>
            <button className="secondary-button" onClick={refresh} disabled={loading}>
              <RefreshCw size={15} /> {loading ? "Loading..." : "Refresh"}
            </button>
          </div>

          <div className="table-controls">
            <div className="search-box">
              <Search size={17} />
              <input
                aria-label="Search tickets"
                placeholder="Search messages, categories or IDs"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>
            <select
              aria-label="Filter ticket status"
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value)}
            >
              <option>All</option>
              <option>Open</option>
              <option>In Progress</option>
              <option>Resolved</option>
            </select>
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>TICKET</th><th>CATEGORY</th><th>PRIORITY</th><th>STATUS</th><th>REVIEW</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((ticket) => (
                  <tr key={ticket.id}>
                    <td>
                      <strong>#{ticket.id}</strong>
                      <p className="ticket-message">{ticket.message}</p>
                    </td>
                    <td><span className="category-chip">{ticket.category}</span></td>
                    <td><span className={`priority-chip ${ticket.priority.toLowerCase()}`}>{ticket.priority}</span></td>
                    <td>
                      <select
                        className="status-select"
                        aria-label={`Status for ticket ${ticket.id}`}
                        value={ticket.status}
                        onChange={(event) => handleStatus(ticket, event.target.value)}
                      >
                        <option>Open</option>
                        <option>In Progress</option>
                        <option>Resolved</option>
                      </select>
                    </td>
                    <td>{ticket.needs_review ? <span className="review-chip">Review</span> : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!loading && filtered.length === 0 && (
              <p className="empty-state">No tickets match this search or filter.</p>
            )}
          </div>
        </section>

        <footer className="workspace-footer">
          ResolveAI · ML-assisted ticket triage · Review uncertain predictions
        </footer>
      </main>
    </div>
  );
}
