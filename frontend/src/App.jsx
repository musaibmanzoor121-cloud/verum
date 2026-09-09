import { useState } from 'react'
import ApplyForm from './components/ApplyForm'
import Dashboard from './components/Dashboard'

// The mark: Verum's logomark — a V-shaped instrument needle taking a reading
// against a measurement baseline with two decision-threshold ticks. It's the
// product's thesis (a reading between thresholds, never a binary stamp) as a
// glyph. Inlined as SVG so it's crisp at any size and needs no network request.
function Mark() {
  return (
    <svg
      className="h-9 w-9"
      viewBox="0 0 512 512"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Verum"
    >
      <title>Verum</title>
      <rect x="16" y="16" width="480" height="480" rx="112" fill="#0B1220" />
      <line x1="140" y1="356" x2="372" y2="356" stroke="#313C4E" strokeWidth="12" strokeLinecap="round" />
      <line x1="218" y1="342" x2="218" y2="370" stroke="#5B6472" strokeWidth="12" strokeLinecap="round" />
      <line x1="294" y1="342" x2="294" y2="370" stroke="#5B6472" strokeWidth="12" strokeLinecap="round" />
      <path d="M150 150 L256 330 L362 150" stroke="#12657A" strokeWidth="46" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="256" cy="330" r="30" fill="#F5F7FA" />
      <circle cx="256" cy="330" r="14" fill="#0E4F5C" />
    </svg>
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
