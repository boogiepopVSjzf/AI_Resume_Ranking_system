import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiFetch } from '../../api/client'
import { useAuth } from '../../auth/AuthContext'
import type { RagSearchResponse } from '../../types/api'

export default function JobSetupPage() {
  const navigate = useNavigate()
  const { token } = useAuth()

  const [hrNote, setHrNote] = useState('')
  const [jdText, setJdText] = useState('')
  const [jdFile, setJdFile] = useState<File | null>(null)
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('hr_note', hrNote)
      formData.append('jd_text', jdText)
      formData.append('top_k', String(topK))

      if (jdFile) {
        formData.append('jd_file', jdFile)
      }

      const data = await apiFetch<RagSearchResponse>(
        '/api/rag-search',
        {
          method: 'POST',
          body: formData,
        },
        token || undefined
      )

      localStorage.setItem('last_rag_result', JSON.stringify(data))
      navigate('/hr/results', { state: { result: data } })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'RAG search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h1>Job Setup</h1>
      <p>Enter HR note and JD, then run candidate retrieval.</p>

      <form onSubmit={handleSubmit}>
        <label className="form-label">HR Note</label>
        <textarea
          className="textarea"
          rows={5}
          value={hrNote}
          onChange={(e) => setHrNote(e.target.value)}
          placeholder="e.g. Need a data engineer with Python, SQL, ETL, and cloud experience"
        />

        <label className="form-label">JD Text</label>
        <textarea
          className="textarea"
          rows={8}
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
          placeholder="Paste job description here..."
        />

        <label className="form-label">JD File (optional)</label>
        <input
          className="input"
          type="file"
          accept=".pdf,.docx"
          onChange={(e) => setJdFile(e.target.files?.[0] || null)}
        />

        <label className="form-label">Top K</label>
        <input
          className="input"
          type="number"
          min={1}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
        />

        <button className="button" type="submit" disabled={loading}>
          {loading ? 'Running...' : 'Run Ranking'}
        </button>

        {error ? <p className="error-text">{error}</p> : null}
      </form>
    </div>
  )
}