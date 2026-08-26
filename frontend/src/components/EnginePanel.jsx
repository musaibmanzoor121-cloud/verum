import ScoreBand from './ScoreBand'

// EnginePanel — a compact card summarizing one detection engine: a kicker
// label, a title, the confidence band, and a few key metrics rendered as a
// clean instrument readout (mono, tabular numbers, hairline row dividers).
export default function EnginePanel({ kicker, title, score, bandLabel, metrics = [], note }) {
  return (
    <div className="rounded-xl2 border border-line bg-panel p-5 shadow-panel">
      <div className="mb-4">
        <div className="font-mono text-[11px] uppercase tracking-widest text-brand/70">
          {kicker}
        </div>
        <h3 className="font-display text-lg text-ink">{title}</h3>
      </div>

      <ScoreBand score={score} bandLabel={bandLabel} variant="compact" />

      {metrics.length > 0 && (
        <dl className="mt-4 border-t border-line pt-1 text-sm">
          {metrics.map((m) => (
            <div
              key={m.label}
              className="flex items-center justify-between gap-4 border-b border-line/60 py-1.5 last:border-0"
            >
              <dt className="text-muted">{m.label}</dt>
              <dd className="text-right font-mono tabular-nums text-ink">{m.value}</dd>
            </div>
          ))}
        </dl>
      )}

      {note && <p className="mt-3 text-xs leading-relaxed text-muted">{note}</p>}
    </div>
  )
}
