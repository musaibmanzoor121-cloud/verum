// api.js — the single place that talks to the FastAPI backend.
// The base URL comes from VITE_API_URL (see .env.example). In development it
// defaults to the local backend.

// Resolve the backend base URL:
//   * If VITE_API_URL is set, use it (e.g. a separate backend host).
//   * Else in a production build, use '' so requests are same-origin
//     (this is how the single-service Docker deploy works).
//   * Else in dev, talk to the local backend on port 8000.
const _raw = import.meta.env.VITE_API_URL
export const API_URL = (
  _raw ?? (import.meta.env.PROD ? '' : 'http://localhost:8000')
).replace(/\/$/, '')

async function handle(res) {
  if (!res.ok) {
    // Try to surface the backend's error message (e.g. the rate-limit notice).
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

// JSON-only path — used when the applicant pastes text. Simple and reliable.
export async function analyzeText({ resume_text, cover_letter_text, behavior }) {
  const res = await fetch(`${API_URL}/api/analyze-text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, cover_letter_text, behavior }),
  })
  return handle(res)
}

// File-upload path — used when the applicant uploads a PDF/DOCX. This also
// exercises the document-metadata signal in the backend.
export async function analyzeFiles({ resumeFile, coverFile, behavior }) {
  const fd = new FormData()
  fd.append('resume', resumeFile)
  if (coverFile) fd.append('cover_letter', coverFile)
  if (behavior) fd.append('behavior', JSON.stringify(behavior))
  const res = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: fd })
  return handle(res)
}
