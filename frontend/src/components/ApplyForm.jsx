import { useEffect, useRef, useState } from 'react'
import { createBehaviorTracker } from '../lib/behaviorTracker'
import { analyzeText, analyzeFiles } from '../api'

// Two ready-made samples so anyone can see the contrast in one click — great
// for demos and screenshots. The "AI" sample is deliberately buzzword-heavy,
// uniform in rhythm, and its cover letter name-drops skills/companies that
// never appear in its resume (which also trips the consistency engine).
const SAMPLE = {
  human: {
    name: 'Priya Nair',
    email: 'priya@example.com',
    resume:
      'Backend engineer, 3 years. At Zappos I rebuilt the returns queue in Go ' +
      'and cut p95 latency from 800ms to 90ms. Shipped it solo over a rough two ' +
      'weeks. Before that, two internships — one at a tiny fintech where I mostly ' +
      'wrote Python glue and learned to hate flaky tests. I care about small teams ' +
      'and shipping. Side project: a Raspberry Pi that texts me when my plants are dry.',
    cover:
      "Hi — I'm applying for the backend role. The returns-queue rewrite at Zappos " +
      "is the thing I'm proudest of; it was ugly Go at first and I rewrote it twice. " +
      "I'm not the flashiest candidate, but I finish things. Happy to walk through " +
      'the latency work on a call.',
  },
  ai: {
    name: 'Alex Morgan',
    email: 'alex@example.com',
    resume:
      'I am a highly results-driven and detail-oriented software professional. ' +
      'I have leveraged cutting-edge solutions to spearhead cross-functional initiatives. ' +
      'I am passionate about delivering best-in-class, robust solutions seamlessly. ' +
      'Furthermore, I have utilized innovative solutions to drive strategic initiatives ' +
      'across dynamic teams. Moreover, I consistently leverage synergy to move the needle.',
    cover:
      'Dear Hiring Manager, I am excited to apply. At Google and Microsoft I leveraged ' +
      'TensorFlow and Kubernetes to spearhead machine learning initiatives. I am a ' +
      'proactive team player passionate about cutting-edge solutions and would be a ' +
      'dynamic addition to your organization.',
  },
}

export default function ApplyForm({ onResult }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [mode, setMode] = useState('paste') // 'paste' | 'upload'
  const [resumeText, setResumeText] = useState('')
  const [coverText, setCoverText] = useState('')
  const [resumeFile, setResumeFile] = useState(null)
  const [coverFile, setCoverFile] = useState(null)
  const [honeypot, setHoneypot] = useState('') // must stay empty for humans
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const trackerRef = useRef(null)

  // Start capturing behavioral biometrics as soon as the form mounts.
  useEffect(() => {
    trackerRef.current = createBehaviorTracker()
    return () => trackerRef.current?.stop()
  }, [])

  function loadSample(kind) {
    const s = SAMPLE[kind]
    setMode('paste')
    setName(s.name)
    setEmail(s.email)
    setResumeText(s.resume)
    setCoverText(s.cover)
    setError('')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const behavior = trackerRef.current?.getPayload(honeypot)
      let report
      if (mode === 'upload') {
        if (!resumeFile) throw new Error('Choose a resume file to upload first.')
        report = await analyzeFiles({ resumeFile, coverFile, behavior })
      } else {
        if (!resumeText.trim())
          throw new Error('Paste your resume text, or load a sample to try it.')
        report = await analyzeText({
          resume_text: resumeText,
          cover_letter_text: coverText,
          behavior,
        })
      }
      onResult(report, { name: name.trim() || 'Anonymous applicant' })
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const field =
    'w-full rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink ' +
    'placeholder:text-muted/60 focus:outline-none focus:ring-2 focus:ring-brand/30'

  return (
    <div className="mx-auto max-w-2xl">
      <header className="mb-6">
        <div className="font-mono text-[11px] uppercase tracking-widest text-brand/70">
          Applicant view
        </div>
        <h2 className="font-display text-2xl text-ink">Apply for this role</h2>
        <p className="mt-1 text-sm text-muted">
          Fill this out like a candidate would. Verum measures the writing and the
          way the form is filled — then shows the recruiter its reasoning.
        </p>
      </header>

      {/* Sample loaders */}
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted">Try a sample:</span>
        <button
          type="button"
          onClick={() => loadSample('human')}
          className="rounded-full border border-line bg-white px-3 py-1 text-xs font-medium text-verify hover:bg-verify/5"
        >
          Human-written
        </button>
        <button
          type="button"
          onClick={() => loadSample('ai')}
          className="rounded-full border border-line bg-white px-3 py-1 text-xs font-medium text-alert hover:bg-alert/5"
        >
          AI-generated
        </button>
      </div>

      <form
        onSubmit={handleSubmit}
        className="rounded-xl2 border border-line bg-panel p-6 shadow-panel"
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-ink">Full name</span>
            <input className={field} value={name} onChange={(e) => setName(e.target.value)} placeholder="Jordan Lee" />
          </label>
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-ink">Email</span>
            <input className={field} type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </label>
        </div>

        {/* Paste / Upload toggle */}
        <div className="mt-5 mb-3 inline-flex rounded-lg border border-line bg-paper p-1 text-sm">
          {['paste', 'upload'].map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={
                'rounded-md px-3 py-1 capitalize transition ' +
                (mode === m ? 'bg-white text-ink shadow-sm' : 'text-muted')
              }
            >
              {m === 'paste' ? 'Paste text' : 'Upload file'}
            </button>
          ))}
        </div>

        {mode === 'paste' ? (
          <div className="space-y-4">
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-ink">Resume</span>
              <textarea
                className={field + ' h-36 resize-y font-sans'}
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste your resume text here…"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-ink">
                Cover letter <span className="text-muted">(optional)</span>
              </span>
              <textarea
                className={field + ' h-28 resize-y'}
                value={coverText}
                onChange={(e) => setCoverText(e.target.value)}
                placeholder="Paste your cover letter (enables the resume↔cover consistency check)…"
              />
            </label>
          </div>
        ) : (
          <div className="space-y-4">
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-ink">Resume file (PDF / DOCX)</span>
              <input type="file" accept=".pdf,.docx,.txt" onChange={(e) => setResumeFile(e.target.files[0])} className="block w-full text-sm text-muted file:mr-3 file:rounded-md file:border-0 file:bg-brand file:px-3 file:py-2 file:text-white" />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-ink">Cover letter file <span className="text-muted">(optional)</span></span>
              <input type="file" accept=".pdf,.docx,.txt" onChange={(e) => setCoverFile(e.target.files[0])} className="block w-full text-sm text-muted file:mr-3 file:rounded-md file:border-0 file:bg-brand file:px-3 file:py-2 file:text-white" />
            </label>
          </div>
        )}

        {/* Honeypot: invisible to humans, catnip for naive bots. */}
        <input
          type="text"
          name="company_website"
          value={honeypot}
          onChange={(e) => setHoneypot(e.target.value)}
          tabIndex={-1}
          autoComplete="off"
          aria-hidden="true"
          className="absolute left-[-9999px] h-0 w-0 opacity-0"
        />

        {error && (
          <p className="mt-4 rounded-lg bg-alert/10 px-3 py-2 text-sm text-alert">{error}</p>
        )}

        <div className="mt-6 flex items-center justify-between gap-4">
          <p className="text-xs text-muted">
            We record cursor/keystroke <em>timing</em> only — never keystroke content.
          </p>
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-brand px-5 py-2.5 text-sm font-semibold text-white shadow-panel transition hover:brightness-110 disabled:opacity-60"
          >
            {loading ? 'Analyzing…' : 'Submit application'}
          </button>
        </div>
      </form>
    </div>
  )
}
