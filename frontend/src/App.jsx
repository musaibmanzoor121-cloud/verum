import { useState } from 'react'
import ApplyForm from './components/ApplyForm'
import Dashboard from './components/Dashboard'

// The mark: a small monogram tile. Kept minimal so the confidence-band meter
// and evidence cards remain the memorable elements, not the logo.
function Mark() {
  return (
    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand font-display text-sm font-bold text-white">
      V
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState('apply') // 'apply' | 'recruiter'
  const [submissions, setSubmissions] = useState([]) // in-memory only
  const [selectedId, setSelectedId] = useState(null)

  function handleResult(report, meta) {
    const entry = { id: crypto.randomUUID(), name: meta.name, report, at: Date.now() }
    setSubmissions((prev) => [entry, ...prev])
    setSelectedId(entry.id)
    setTab('recruiter') // flip to the recruiter view to reveal the report
  }

  const TabButton = ({ id, children, badge }) => (
    <button
      onClick={() => setTab(id)}
      className={
        'relative rounded-md px-3.5 py-1.5 text-sm font-medium transition ' +
        (tab === id ? 'bg-white text-ink shadow-sm' : 'text-muted hover:text-ink')
      }
    >
      {children}
      {badge > 0 && (
        <span className="ml-1.5 rounded-full bg-brand px-1.5 py-0.5 font-mono text-[10px] text-white">
          {badge}
        </span>
      )}
    </button>
  )

  return (
    <div className="min-h-full">
      {/* Top bar */}
      <header className="sticky top-0 z-10 border-b border-line bg-paper/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <div className="flex items-center gap-3">
            <Mark />
            <div className="leading-tight">
              <div className="font-display text-lg font-bold text-ink">Verum</div>
              <div className="text-[11px] text-muted">Applicant Authenticity Engine</div>
            </div>
          </div>
          <nav className="inline-flex rounded-lg border border-line bg-paper p-1">
            <TabButton id="apply">Applicant</TabButton>
            <TabButton id="recruiter" badge={submissions.length}>Recruiter</TabButton>
          </nav>
        </div>
      </header>

      {/* Body */}
      <main className="mx-auto max-w-6xl px-5 py-8">
        {tab === 'apply' ? (
          <ApplyForm onResult={handleResult} />
        ) : (
          <Dashboard submissions={submissions} selectedId={selectedId} onSelect={setSelectedId} />
        )}
      </main>

      {/* Footer — states the product's ethic in its own voice. */}
      <footer className="mx-auto max-w-6xl px-5 pb-10 pt-4">
        <p className="border-t border-line pt-4 text-xs text-muted">
          Verum flags applications for <strong className="text-ink">human review</strong> — it
          never auto-rejects. Scores are confidence bands, not verdicts, and every flag shows
          its evidence.
        </p>
      </footer>
    </div>
  )
}
