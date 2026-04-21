import { useState } from 'react'
import { apiFetch } from '../../api/client'
import { useAuth } from '../../auth/AuthContext'
import type { ParseBatchResponse } from '../../types/api'

export default function ResumeImportPage() {
  const { token } = useAuth()

  const [files, setFiles] = useState<FileList | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<ParseBatchResponse | null>(null)

  async function handleUpload() {
    if (!files || files.length === 0) {
      setError('Please choose at least one file first.')
      return
    }

    setError('')
    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()

      Array.from(files).forEach((file) => {
        formData.append('files', file)
      })

      const data = await apiFetch<ParseBatchResponse>(
        '/api/parse/batch',
        {
          method: 'POST',
          body: formData,
        },
        token || undefined
      )

      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h1>Resume Import</h1>
      <p>Upload multiple resumes for parsing and storage.</p>

      <input
        className="input"
        type="file"
        multiple
        accept=".pdf,.docx"
        onChange={(e) => setFiles(e.target.files)}
      />

      <button className="button" onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload Resumes'}
      </button>

      {error ? <p className="error-text">{error}</p> : null}

      {result ? (
        <div className="result-block">
          <h3>Batch Result</h3>
          <p>Total: {result.total}</p>
          <p>Succeeded: {result.succeeded_count}</p>
          <p>Failed: {result.failed_count}</p>

          <h4>Succeeded</h4>
          <pre className="json-box">{JSON.stringify(result.succeeded, null, 2)}</pre>

          <h4>Failed</h4>
          <pre className="json-box">{JSON.stringify(result.failed, null, 2)}</pre>
        </div>
      ) : null}
    </div>
  )
}