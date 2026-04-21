import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { apiFetch } from '../api/client'
import type { LoginResponse, MeResponse, UserRole } from '../types/auth'

interface AuthState {
  token: string | null
  username: string | null
  role: UserRole | null
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | undefined>(undefined)

const TOKEN_KEY = 'auth_token'
const USERNAME_KEY = 'auth_username'
const ROLE_KEY = 'auth_role'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(localStorage.getItem(TOKEN_KEY))
  const [username, setUsername] = useState<string | null>(localStorage.getItem(USERNAME_KEY))
  const [role, setRole] = useState<UserRole | null>(
    (localStorage.getItem(ROLE_KEY) as UserRole | null) || null
  )

  useEffect(() => {
    async function restoreSession() {
      if (!token) return

      try {
        const me = await apiFetch<MeResponse>('/auth/me', { method: 'GET' }, token)
        setUsername(me.username)
        setRole(me.role)
        localStorage.setItem(USERNAME_KEY, me.username)
        localStorage.setItem(ROLE_KEY, me.role)
      } catch {
        logout()
      }
    }

    restoreSession()
  }, [token])

  async function login(usernameInput: string, password: string) {
    const data = await apiFetch<LoginResponse>('/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: usernameInput,
        password,
      }),
    })

    setToken(data.access_token)
    setUsername(data.username)
    setRole(data.role)

    localStorage.setItem(TOKEN_KEY, data.access_token)
    localStorage.setItem(USERNAME_KEY, data.username)
    localStorage.setItem(ROLE_KEY, data.role)
  }

  function logout() {
    setToken(null)
    setUsername(null)
    setRole(null)

    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USERNAME_KEY)
    localStorage.removeItem(ROLE_KEY)
  }

  const value = useMemo(
    () => ({
      token,
      username,
      role,
      isAuthenticated: !!token,
      login,
      logout,
    }),
    [token, username, role]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}