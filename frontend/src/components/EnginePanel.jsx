import ScoreBand from './ScoreBand'

// EnginePanel — a compact card summarizing one detection engine: a kicker
// label, a title, the confidence band, and a few key metrics.
export default function EnginePanel({ kicker, title, score, bandLabel, metrics = [], note }) {
  return (
    <div className="rounded-xl2 border border-line bg-panel p-5 shadow-panel">
      <div className="mb-4">
        <div className="font-mono text-[11px] uppercase tracking-widest text-brand/70">
          {kicker}
        </div>
        <h3 className="font-display text-lg text-ink">{title}</h3>
      </div>

      <ScoreBand score={score} bandLabel={bandLabel} />

      {metrics.length > 0 && (
        <dl className="mt-4 space-y-1.5 border-t border-line pt-3">
          {metrics.map((m) => (
            <div key={m.label} className="flex justify-between gap-4 text-sm">
              <dt className="text-muted">{m.label}</dt>
              <dd className="font-mono text-ink text-right">{m.value}</dd>
            </div>
          ))}
        </dl>
      )}

      {note && <p className="mt-3 text-xs text-muted leading-relaxed">{note}</p>}
    </div>
  )
}
