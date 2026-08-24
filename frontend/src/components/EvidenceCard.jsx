import { severityTone, TONE } from '../lib/ui'

const ENGINE_LABEL = {
  content: 'Content',
  behavior: 'Behavior',
  consistency: 'Consistency',
  metadata: 'Metadata',
}

// EvidenceCard — one "receipt" explaining a single flag. This is the core
// differentiator: never a bare number. Each card names the engine + signal,
// explains it in plain language, and (when relevant) shows the EXACT excerpt
// that triggered it, highlighted like a marked-up document.
export default function EvidenceCard({ item }) {
  const t = TONE[severityTone(item.severity)]
  return (
    <div className="rise-in rounded-xl border border-line bg-panel p-4 shadow-panel">
      <div className="flex items-center gap-2 mb-1.5">
        <span
          className={`text-[10px] font-mono uppercase tracking-wide px-2 py-0.5 rounded-full ${t.soft} ${t.text}`}
        >
          {item.severity}
        </span>
        <span className="text-xs font-medium text-muted">
          {ENGINE_LABEL[item.engine] || item.engine}
        </span>
        <span className="ml-auto font-mono text-[11px] text-muted">{item.signal}</span>
      </div>
      <p className="text-sm text-ink leading-relaxed">{item.detail}</p>
      {item.excerpt && (
        <blockquote className="mt-2.5 text-sm leading-relaxed">
          <span className="evidence-mark">{item.excerpt}</span>
        </blockquote>
      )}
    </div>
  )
}
