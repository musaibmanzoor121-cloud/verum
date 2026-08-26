import { toneFor, TONE } from '../lib/ui'

// ScoreBand — Verum's signature instrument.
//
// Verum's whole thesis is "confidence BANDS between decision THRESHOLDS, never
// a fake-precise verdict." So the meter is built to *teach that structure at a
// glance* instead of decorating a number:
//
//   • the track is split into the three real verdict ZONES —
//       authentic (0–34) · review (34–66) · strong (66–100)
//   • the two decision THRESHOLDS (34, 66) are drawn as ticks
//   • the score is shown as a shaded RANGE (center ± 10), not a single needle,
//     because the model reports a band, not a point
//
// variant="hero"    → full zoned instrument with labels + big readout (report header)
// variant="compact" → the same range logic, quiet, for the per-engine panels
export default function ScoreBand({
  score = 0,
  title = 'Concern level',
  bandLabel,
  variant = 'compact',
}) {
  const center = Math.round((score || 0) * 100)
  const lo = Math.max(0, center - 10)
  const hi = Math.min(100, center + 10)
  const t = TONE[toneFor(score)]
  const zone =
    score < 0.34 ? 'authentic' : score < 0.66 ? 'needs review' : 'strong signal'

  if (variant === 'hero') {
    return (
      <div>
        <div className="mb-2 flex items-end justify-between">
          <span className="text-sm font-medium text-muted">{title}</span>
          <div className="text-right">
            <span className={`font-display text-3xl font-bold tabular-nums ${t.text}`}>
              {center}
              <span className="text-lg font-medium">%</span>
            </span>
            <div className="font-mono text-[11px] text-muted">
              band {lo}–{hi}%
            </div>
          </div>
        </div>

        {/* The instrument */}
        <div
          className="relative h-11 overflow-hidden rounded-lg ring-1 ring-line"
          role="img"
          aria-label={`${title}: ${center} percent, ${zone}. Confidence band ${lo} to ${hi} percent.`}
        >
          {/* Verdict zones (faint) */}
          <div className="absolute inset-y-0 left-0 bg-verify/10" style={{ width: '34%' }} />
          <div className="absolute inset-y-0 bg-review/10" style={{ left: '34%', width: '32%' }} />
          <div className="absolute inset-y-0 bg-alert/10" style={{ left: '66%', width: '34%' }} />

          {/* Threshold ticks at the real decision boundaries */}
          <div className="absolute inset-y-0 w-px bg-line" style={{ left: '34%' }} />
          <div className="absolute inset-y-0 w-px bg-line" style={{ left: '66%' }} />

          {/* The confidence RANGE (center ± 10) */}
          <div
            className={`absolute inset-y-1.5 ${t.bg} rounded-md shadow-sm transition-all duration-700 ease-out`}
            style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 4)}%` }}
            aria-hidden="true"
          />
          {/* Center marker */}
          <div
            className="absolute inset-y-0 w-0.5 bg-ink/70 transition-all duration-700 ease-out"
            style={{ left: `calc(${center}% - 1px)` }}
            aria-hidden="true"
          />
        </div>

        {/* Zone legend, aligned to the thresholds */}
        <div className="relative mt-1.5 h-4 font-mono text-[10px] uppercase tracking-wide text-muted">
          <span className="absolute left-0">authentic</span>
          <span className="absolute -translate-x-1/2" style={{ left: '50%' }}>
            review
          </span>
          <span className="absolute right-0">strong</span>
        </div>
      </div>
    )
  }

  // compact — quiet supporting meter for engine panels
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-medium text-muted">{title}</span>
        {bandLabel && <span className={`font-mono text-xs ${t.text}`}>{bandLabel}</span>}
      </div>
      <div className="band-track relative h-3 overflow-hidden rounded-full ring-1 ring-line">
        {/* threshold ticks, kept subtle */}
        <div className="absolute inset-y-0 z-10 w-px bg-line/80" style={{ left: '34%' }} />
        <div className="absolute inset-y-0 z-10 w-px bg-line/80" style={{ left: '66%' }} />
        <div
          className={`absolute inset-y-0 ${t.bg} rounded-full transition-all duration-700 ease-out`}
          style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 3)}%` }}
          aria-hidden="true"
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted">
        <span>0</span>
        <span>50</span>
        <span>100</span>
      </div>
    </div>
  )
}
