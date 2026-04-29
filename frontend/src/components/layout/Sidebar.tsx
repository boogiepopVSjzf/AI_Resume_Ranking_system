import { NAV_ITEMS, type RouteId } from "../../lib/constants";

type SidebarProps = {
  activeRoute: RouteId;
  onNavigate: (route: RouteId) => void;
};

export function Sidebar({ activeRoute, onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <a className="brand-mark" href="#top" aria-label="Entagile home">
          <img src="/entagile-logo.png" alt="entagile" width={155} height={69} />
        </a>
        <div className="brand-block-text">
          <p className="brand-overline">Recruiting suite</p>
          <h2>Resume intelligence</h2>
        </div>
      </div>

      <nav className="nav-list" aria-label="Primary navigation">
        {NAV_ITEMS.map((item, index) => (
          <button
            key={item.id}
            className={`nav-item ${activeRoute === item.id ? "active" : ""}`}
            onClick={() => onNavigate(item.id)}
            type="button"
          >
            <span className="nav-index">{String(index + 1).padStart(2, "0")}</span>
            <span>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <p>Backend</p>
        <strong>Online</strong>
        <p className="sidebar-credit">Powered by Entagile</p>
      </div>
    </aside>
  );
}
