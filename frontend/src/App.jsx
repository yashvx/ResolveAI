import { useState } from "react";
import {
  ArrowRight,
  BrainCircuit,
  ChartNoAxesCombined,
  Clock3,
  Menu,
  MessageSquareText,
  ShieldCheck,
  Sparkles,
  Ticket,
  X,
} from "lucide-react";
import Dashboard from "./Dashboard";
import "./App.css";

const features = [
  {
    icon: BrainCircuit,
    title: "AI-powered classification",
    description:
      "Categorize support issues with a machine learning text classifier.",
  },
  {
    icon: Clock3,
    title: "Smarter prioritization",
    description:
      "Surface potentially urgent problems using transparent priority rules.",
  },
  {
    icon: ChartNoAxesCombined,
    title: "Actionable analytics",
    description:
      "Explore ticket categories, status counts, and support workload.",
  },
];

function App() {
  const [showDashboard, setShowDashboard] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  function openDashboard() {
    setShowDashboard(true);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  if (showDashboard) {
    return <Dashboard onBack={() => setShowDashboard(false)} />;
  }

  return (
    <div className="app-shell">
      <header className="navbar">
        <a className="brand" href="#home">
          <span className="brand-icon"><BrainCircuit size={23} /></span>
          <span>Resolve<span className="brand-accent">AI</span></span>
        </a>

        <button
          className="mobile-menu-button"
          aria-label={menuOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X /> : <Menu />}
        </button>

        <nav className={menuOpen ? "nav-links open" : "nav-links"}>
          <a href="#features" onClick={() => setMenuOpen(false)}>Features</a>
          <a href="#how-it-works" onClick={() => setMenuOpen(false)}>How it works</a>
          <a href="#product" onClick={() => setMenuOpen(false)}>Product preview</a>
          <button className="button button-dark nav-cta" onClick={openDashboard}>
            Open dashboard <ArrowRight size={16} />
          </button>
        </nav>
      </header>

      <main>
        <section className="hero section-container" id="home">
          <div className="hero-copy">
            <div className="announcement">
              <Sparkles size={15} />
              <span>Intelligent support operations</span>
            </div>
            <h1>Turn support chaos into <span>clarity.</span></h1>
            <p className="hero-description">
              ResolveAI helps support teams classify customer issues,
              identify uncertain predictions, and prioritize their workload
              with machine learning and explainable rules.
            </p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={openDashboard}>
                Explore the platform <ArrowRight size={17} />
              </button>
              <a className="text-link" href="#how-it-works">See how it works</a>
            </div>
            <div className="trust-line">
              <ShieldCheck size={17} />
              <span>Human review for uncertain predictions.</span>
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-glow" />
            <div className="floating-label label-top">
              <span className="status-dot" /> ML-powered analysis
            </div>
            <div className="dashboard-preview">
              <div className="preview-sidebar">
                <div className="preview-brand"><BrainCircuit size={21} /> ResolveAI</div>
                <div className="preview-nav active"><ChartNoAxesCombined size={17} /> Overview</div>
                <div className="preview-nav"><Ticket size={17} /> All tickets</div>
                <div className="preview-nav"><BrainCircuit size={17} /> AI analyzer</div>
              </div>
              <div className="preview-content">
                <div className="preview-topbar">
                  <div>
                    <span className="eyebrow">WORKSPACE / OVERVIEW</span>
                    <h3>Support overview</h3>
                  </div>
                  <span className="avatar">RA</span>
                </div>
                <div className="metrics-grid">
                  <div className="metric-card"><span>Total tickets</span><strong>Live</strong><small>Database-backed</small></div>
                  <div className="metric-card"><span>AI analysis</span><strong>ML</strong><small>Text classification</small></div>
                  <div className="metric-card"><span>Ticket status</span><strong>Track</strong><small>Update in dashboard</small></div>
                </div>
                <div className="ticket-panel">
                  <div className="panel-heading">
                    <div><h4>Support workspace</h4><p>Analyze and manage customer issues</p></div>
                    <span className="sample-badge">LIVE APP</span>
                  </div>
                  <div className="ticket-row"><span className="ticket-id">01</span><span className="ticket-name">Submit a new customer ticket</span></div>
                  <div className="ticket-row"><span className="ticket-id">02</span><span className="ticket-name">Review predicted categories</span></div>
                  <div className="ticket-row"><span className="ticket-id">03</span><span className="ticket-name">Monitor ticket analytics</span></div>
                </div>
              </div>
            </div>
            <div className="floating-label label-bottom">
              <ShieldCheck size={17} /> Human-reviewed recommendations
            </div>
          </div>
        </section>

        <section className="proof-strip">
          <div className="section-container proof-inner">
            <span>ONE WORKSPACE</span>
            <span>ML CLASSIFICATION</span>
            <span>EXPLAINABLE PRIORITY RULES</span>
            <span>LIVE TICKET ANALYTICS</span>
          </div>
        </section>

        <section className="section-container section-block" id="features">
          <div className="section-heading">
            <span className="eyebrow">WHAT RESOLVEAI DOES</span>
            <h2>Less manual sorting. <span>More meaningful action.</span></h2>
            <p>Turn unstructured customer messages into useful, reviewable information.</p>
          </div>
          <div className="feature-grid">
            {features.map(({ icon: Icon, title, description }, index) => (
              <article className="feature-card" key={title}>
                <span className="feature-icon"><Icon size={23} /></span>
                <span className="feature-number">0{index + 1}</span>
                <h3>{title}</h3>
                <p>{description}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="workflow-section" id="how-it-works">
          <div className="section-container workflow-inner">
            <div className="workflow-copy">
              <span className="eyebrow">A SIMPLE WORKFLOW</span>
              <h2>From raw ticket to <span>informed action.</span></h2>
              <p>Move from a customer complaint to a reviewable recommendation through one clear workflow.</p>
              <button className="button button-primary" onClick={openDashboard}>
                Try the dashboard <ArrowRight size={16} />
              </button>
            </div>
            <div className="workflow-steps">
              <div className="workflow-step"><span className="step-icon"><MessageSquareText /></span><div><strong>01. Capture</strong><p>Submit a customer issue.</p></div></div>
              <div className="workflow-step"><span className="step-icon"><BrainCircuit /></span><div><strong>02. Analyze</strong><p>Predict a category with ML.</p></div></div>
              <div className="workflow-step"><span className="step-icon"><ShieldCheck /></span><div><strong>03. Review</strong><p>Flag uncertain predictions for review.</p></div></div>
              <div className="workflow-step"><span className="step-icon"><ChartNoAxesCombined /></span><div><strong>04. Improve</strong><p>Track ticket trends and status.</p></div></div>
            </div>
          </div>
        </section>

        <section className="section-container product-section" id="product">
          <div className="product-cta">
            <span className="eyebrow">YOUR SUPPORT WORKSPACE</span>
            <h2>See the whole picture. <span>Take the next step.</span></h2>
            <p>Analyze tickets, manage statuses, and explore metrics from your own database.</p>
            <button className="button button-primary" onClick={openDashboard}>
              Open live dashboard <ArrowRight size={17} />
            </button>
          </div>
          <div className="product-note">
            <span className="feature-icon"><Ticket size={24} /></span>
            <h3>Connected to your backend</h3>
            <p>The dashboard fetches ticket records and analytics from your local FastAPI service.</p>
            <button className="button button-dark" onClick={openDashboard}>Launch ResolveAI <ArrowRight size={16} /></button>
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="section-container footer-inner">
          <a className="brand footer-brand" href="#home">
            <span className="brand-icon"><BrainCircuit size={21} /></span>
            <span>Resolve<span className="brand-accent">AI</span></span>
          </a>
          <p>Making customer support operations more intelligent.</p>
          <span className="footer-note">AIML project MVP · 2026</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
