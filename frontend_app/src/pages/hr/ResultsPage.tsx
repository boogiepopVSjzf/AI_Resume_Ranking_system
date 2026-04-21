import { Link, useLocation } from 'react-router-dom'
import type { RagSearchResponse } from '../../types/api'

type LocationState = {
  result?: RagSearchResponse
}

export default function ResultsPage() {
  const location = useLocation()
  const state = location.state as LocationState | null

  const resultFromState = state?.result
  const resultFromStorage = localStorage.getItem('last_rag_result')
  const result: RagSearchResponse | null =
    resultFromState ||
    (resultFromStorage ? (JSON.parse(resultFromStorage) as RagSearchResponse) : null)

  if (!result) {
    return (
      <div className="card">
        <h1>Results</h1>
        <p>No result found yet. Please run Job Setup first.</p>
        <Link className="button-link" to="/hr">
          Go to Job Setup
        </Link>
      </div>
    )
  }

  return (
    <div className="card">
      <h1>Results</h1>

      <div className="result-block">
        <h3>Search Query</h3>
        <p>{result.search_query || '(empty)'}</p>
      </div>

      <div className="result-block">
        <h3>Count</h3>
        <p>{result.count}</p>
      </div>

      <div className="result-block">
        <h3>Top K Returned Resume IDs</h3>
        {result.top_k_resume_ids.length === 0 ? (
          <p>No matched resumes.</p>
        ) : (
          <ul className="result-list">
            {result.top_k_resume_ids.map((id) => (
              <li key={id}>{id}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="result-block">
        <h3>Filtered Resume IDs</h3>
        {result.filtered_resume_ids.length === 0 ? (
          <p>No filtered resumes.</p>
        ) : (
          <ul className="result-list">
            {result.filtered_resume_ids.map((id) => (
              <li key={id}>{id}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="result-block">
        <h3>Hard Filters</h3>
        <pre className="json-box">{JSON.stringify(result.hard_filters, null, 2)}</pre>
      </div>
    </div>
  )
}