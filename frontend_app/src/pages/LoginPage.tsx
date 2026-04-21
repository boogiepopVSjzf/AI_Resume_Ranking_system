import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      await login(username, password)

      if (username === 'hr_demo') {
        navigate('/hr')
      } else {
        navigate('/internal')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-center">
      <form className="card login-card" onSubmit={handleLogin}>
        <h1>AI Resume Ranking</h1>
        <p>Please sign in</p>

        <input
          className="input"
          placeholder="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <input
          className="input"
          type="password"
          placeholder="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button className="button" type="submit" disabled={loading}>
          {loading ? 'Signing in...' : 'Login'}
        </button>

        {error ? <p className="error-text">{error}</p> : null}

        <div className="hint-box">
          <div>HR: hr_demo / hr123456</div>
          <div>Internal: internal_demo / internal123456</div>
        </div>
      </form>
    </div>
  )
}