import { toneFor, TONE } from '../lib/ui'

// ScoreBand — Verum's signature meter.
// Instead of a single needle pointing at a fake-precise number, it renders a
// shaded RANGE (center ± 10 points), visually reinforcing that every score is
// a confidence *band*. The fill color reflects the concern level.
export default function ScoreBand({ score = 0, title = 'Concern level', bandLabel }) {
  const center = Math.round((score || 0) * 100)
  const lo = Math.max(0, center - 10)
  const hi = Math.min(100, center + 10)
  const t = TONE[toneFor(score)]

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-sm font-medium text-muted">{title}</span>
        {bandLabel && <span className={`font-mono text-xs ${t.text}`}>{bandLabel}</span>}
      </div>
      <div className="relative h-3 rounded-full band-track overflow-hidden ring-1 ring-line">
        <div
          className={`absolute inset-y-0 ${t.bg} rounded-full transition-all duration-700 ease-out`}
          style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 3)}%` }}
          aria-hidden="true"
        />
      </div>
      <div className="flex justify-between mt-1 text-[10px] font-mono text-muted">
        <span>0</span>
        <span>50</span>
        <span>100</span>
      </div>
    </div>
  )
}
