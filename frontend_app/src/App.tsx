import { Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import ProtectedRoute from './auth/ProtectedRoute'
import LoginPage from './pages/LoginPage'
import HRLayout from './layouts/HRLayout'
import InternalLayout from './layouts/InternalLayout'
import JobSetupPage from './pages/hr/JobSetupPage'
import ResultsPage from './pages/hr/ResultsPage'
import ResumeImportPage from './pages/internal/ResumeImportPage'
import SchemaPage from './pages/internal/SchemaPage'

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />

        <Route
          path="/hr"
          element={
            <ProtectedRoute allow={['hr', 'internal']}>
              <HRLayout />
            </ProtectedRoute>
          }
        >
          <Route index element={<JobSetupPage />} />
          <Route path="results" element={<ResultsPage />} />
        </Route>

        <Route
          path="/internal"
          element={
            <ProtectedRoute allow={['internal']}>
              <InternalLayout />
            </ProtectedRoute>
          }
        >
          <Route index element={<ResumeImportPage />} />
          <Route path="schema" element={<SchemaPage />} />
        </Route>

        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </AuthProvider>
  )
}