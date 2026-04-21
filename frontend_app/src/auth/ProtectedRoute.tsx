import { Navigate } from 'react-router-dom'
import type { ReactNode } from 'react'
import { useAuth } from './AuthContext'
import type { UserRole } from '../types/auth'

interface ProtectedRouteProps {
  children: ReactNode
  allow: UserRole[]
}

export default function ProtectedRoute({ children, allow }: ProtectedRouteProps) {
  const { isAuthenticated, role } = useAuth()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  if (!role || !allow.includes(role)) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}