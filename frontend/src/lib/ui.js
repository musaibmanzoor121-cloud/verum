// lib/ui.js — tiny shared helpers for consistent color language across the UI.
// NOTE: class names are written as full literal strings (not built dynamically)
// so Tailwind's compiler can see and include them.

export function toneFor(score) {
  if (score == null) return 'verify'
  if (score < 0.34) return 'verify'
  if (score < 0.66) return 'review'
  return 'alert'
}

export function severityTone(sev) {
  if (sev === 'high') return 'alert'
  if (sev === 'medium') return 'review'
  return 'verify'
}

export const TONE = {
  verify: { text: 'text-verify', bg: 'bg-verify', soft: 'bg-verify/10' },
  review: { text: 'text-review', bg: 'bg-review', soft: 'bg-review/10' },
  alert: { text: 'text-alert', bg: 'bg-alert', soft: 'bg-alert/10' },
}
