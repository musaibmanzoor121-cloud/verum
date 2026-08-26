import { severityTone, TONE } from '../lib/ui'

const ENGINE_LABEL = {
  content: 'Content',
  behavior: 'Behavior',
  consistency: 'Consistency',
  metadata: 'Metadata',
}

// EvidenceCard — one "receipt" explaining a single flag. This is the core
// differentiator: never a bare number. Each card carries a severity-colored
// spine, names the engine + signal, explains it in plain language, and (when
// relevant) shows the EXACT excerpt that triggered it, highlighted like a
// document that's been marked up by hand.
export default function EvidenceCard({ item }) {
  const t = TONE[severityTone(item.severity)]
  return (
    <div className="rise-in relative overflow-hidden rounded-xl border border-line bg-panel p-4 pl-5 shadow-panel">
      {/* Severity spine — the receipt's color-coded edge */}
      <span className={`absolute inset-y-0 left-0 w-1 ${t.bg}`} aria-hidden="true" />

      <div className="mb-1.5 flex items-center gap-2">
        <span
          className={`rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide ${t.soft} ${t.text}`}
        >
          {item.severity}
        </span>
        <span className="text-xs font-medium text-muted">
          {ENGINE_LABEL[item.engine] || item.engine}
        </span>
        <span className="ml-auto font-mono text-[11px] text-muted">{item.signal}</span>
      </div>

      <p className="text-sm leading-relaxed text-ink">{item.detail}</p>

      {item.excerpt && (
        <blockquote className="mt-2.5 border-l-2 border-line pl-3 text-sm leading-relaxed">
          <span className="evidence-mark">{item.excerpt}</span>
        </blockquote>
      )}
    </div>
  )
}
