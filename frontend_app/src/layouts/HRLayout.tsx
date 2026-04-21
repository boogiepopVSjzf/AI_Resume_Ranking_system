import { Link, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function HRLayout() {
  const { username, logout } = useAuth()

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>HR Portal</h2>
        <p className="sidebar-user">{username}</p>

        <nav className="nav">
          <Link to="/hr">Job Setup</Link>
          <Link to="/hr/results">Results</Link>
        </nav>

        <button className="button secondary" onClick={logout}>
          Logout
        </button>
      </aside>

      <main className="content">
        <Outlet />
      </main>
    </div>
  )
}