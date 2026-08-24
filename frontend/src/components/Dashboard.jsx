import EnginePanel from './EnginePanel'
import EvidenceCard from './EvidenceCard'
import ScoreBand from './ScoreBand'
import { toneFor, TONE } from '../lib/ui'

const pct = (x) => `${Math.round((x || 0) * 100)}%`
const yn = (b) => (b ? 'Yes' : 'No')

function VerdictChip({ score, children }) {
  const t = TONE[toneFor(score)]
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${t.soft} ${t.text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${t.bg}`} />
      {children}
    </span>
  )
}

// Empty state — an invitation to act, in the interface's own voice.
function EmptyState() {
  return (
    <div className="rounded-xl2 border border-dashed border-line bg-panel/60 p-12 text-center">
      <div className="font-display text-lg text-ink">No applications analyzed yet</div>
      <p className="mx-auto mt-1 max-w-sm text-sm text-muted">
        Switch to the <strong>Applicant view</strong>, submit an application (try a
        sample), and its authenticity report will appear here.
      </p>
    </div>
  )
}

export default function Dashboard({ submissions, selectedId, onSelect }) {
  if (submissions.length === 0) return <EmptyState />

  const current = submissions.find((s) => s.id === selectedId) || submissions[0]
  const r = current.report
  const c = r.content
  const b = r.behavior
  const cons = r.consistency

  return (
    <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
      {/* Left rail: submission list */}
      <aside className="space-y-2">
        <div className="px-1 font-mono text-[11px] uppercase tracking-widest text-brand/70">
          Queue · {submissions.length}
        </div>
        {submissions.map((s) => {
          const active = s.id === current.id
          return (
            <button
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={
                'block w-full rounded-xl border p-3 text-left transition ' +
                (active
                  ? 'border-brand/40 bg-white shadow-panel'
                  : 'border-line bg-panel hover:border-brand/20')
              }
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-medium text-ink">{s.name}</span>
                <span className="font-mono text-[11px] text-muted">{pct(s.report.overall_flag_score)}</span>
              </div>
              <div className="mt-1.5">
                <VerdictChip score={s.report.overall_flag_score}>{s.report.verdict}</VerdictChip>
              </div>
            </button>
          )
        })}
      </aside>

      {/* Detail */}
      <section className="space-y-6">
        {/* Verdict header */}
        <div className="rounded-xl2 border border-line bg-panel p-6 shadow-panel">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="font-mono text-[11px] uppercase tracking-widest text-brand/70">
                Authenticity report · {current.name}
              </div>
              <h2 className="font-display text-2xl text-ink">{r.verdict}</h2>
              <p className="mt-1 max-w-xl text-sm text-muted">{r.recommendation}</p>
            </div>
            <VerdictChip score={r.overall_flag_score}>
              Overall concern {pct(r.overall_flag_score)}
            </VerdictChip>
          </div>
          <div className="mt-5 max-w-md">
            <ScoreBand score={r.overall_flag_score} title="Overall concern" />
          </div>
          <p className="mt-3 font-mono text-[11px] text-muted">
            Processed in {r.processing_ms} ms · perplexity method: {c.perplexity_method}
          </p>
        </div>

        {/* Engine panels */}
        <div className="grid gap-4 md:grid-cols-3">
          <EnginePanel
            kicker="Engine A"
            title="Content authenticity"
            score={c.ai_likelihood_score}
            bandLabel={c.ai_likelihood_band}
            metrics={[
              { label: 'Perplexity', value: c.perplexity ?? `n/a (${c.perplexity_method})` },
              { label: 'Burstiness', value: c.burstiness },
              { label: 'Buzzwords / 100w', value: c.buzzword_density },
              { label: 'Metadata flags', value: c.metadata_flags.length },
            ]}
          />
          <EnginePanel
            kicker="Engine B"
            title="Bot & behavior"
            score={b.bot_likelihood_score}
            bandLabel={b.bot_likelihood_band}
            metrics={[
              { label: 'Time to submit', value: b.time_to_submit_seconds != null ? `${b.time_to_submit_seconds}s` : '—' },
              { label: 'Honeypot filled', value: yn(b.honeypot_triggered) },
              { label: 'Suspicious UA', value: yn(b.suspicious_user_agent) },
              { label: 'Mouse entropy', value: b.mouse_entropy ?? '—' },
              { label: 'Keystroke regularity', value: b.keystroke_regularity ?? '—' },
            ]}
          />
          <EnginePanel
            kicker="Cross-doc"
            title="Resume ↔ cover"
            score={cons.checked ? 1 - cons.consistency_score : 0}
            bandLabel={cons.checked ? `${pct(cons.consistency_score)} aligned` : 'not checked'}
            metrics={[
              { label: 'Alignment', value: cons.checked ? pct(cons.consistency_score) : '—' },
              { label: 'Mismatches', value: cons.mismatches.length },
            ]}
            note={!cons.checked ? 'No cover letter provided, so no cross-document check was run.' : undefined}
          />
        </div>

        {/* Evidence feed — the differentiator */}
        <div>
          <div className="mb-3 flex items-baseline justify-between">
            <h3 className="font-display text-lg text-ink">Evidence — why this was flagged</h3>
            <span className="font-mono text-[11px] text-muted">{r.evidence.length} item(s)</span>
          </div>
          {r.evidence.length === 0 ? (
            <div className="rounded-xl border border-verify/30 bg-verify/5 p-4 text-sm text-verify">
              No flags raised. Nothing in the writing or submission behavior looked automated.
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2">
              {r.evidence.map((item, i) => (
                <EvidenceCard key={i} item={item} />
              ))}
            </div>
          )}
        </div>

        {/* Honest limitations note */}
        <p className="rounded-xl border border-line bg-paper px-4 py-3 text-xs leading-relaxed text-muted">
          <strong className="text-ink">On using these results:</strong> {r.limitations}
        </p>
      </section>
    </div>
  )
}
