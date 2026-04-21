import { Link, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function InternalLayout() {
  const { username, logout } = useAuth()

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>Internal Portal</h2>
        <p className="sidebar-user">{username}</p>

        <nav className="nav">
          <Link to="/internal">Resume Import</Link>
          <Link to="/internal/schema">Schema Library</Link>
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