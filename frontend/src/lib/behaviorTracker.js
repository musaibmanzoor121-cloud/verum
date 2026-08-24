// lib/behaviorTracker.js
// =======================
// Captures the *behavioral biometrics* that power Engine B. While the applicant
// fills the form we quietly record:
//   * mouse movement samples (throttled) -> movement-entropy signal
//   * inter-keystroke timing (NOT which keys — privacy) -> typing-rhythm signal
//   * scroll timestamps
// On submit we also read the honeypot field and the elapsed time.
//
// Nothing here identifies the user or captures content; it only measures the
// *shape* of how the interaction happened, which is what distinguishes a human
// from a script.

export function createBehaviorTracker() {
  const loadTime = Date.now()
  const mouse = []
  const keyIntervals = []
  const scrolls = []
  let lastKey = null
  let lastSample = 0

  const onMove = (e) => {
    const now = Date.now()
    if (now - lastSample < 45) return // throttle to ~22 samples/sec
    lastSample = now
    mouse.push({ t: now - loadTime, x: e.clientX, y: e.clientY })
    if (mouse.length > 500) mouse.shift() // cap memory
  }

  const onKey = () => {
    const now = Date.now()
    if (lastKey !== null) keyIntervals.push(now - lastKey)
    lastKey = now
  }

  const onScroll = () => scrolls.push(Date.now() - loadTime)

  window.addEventListener('mousemove', onMove)
  window.addEventListener('keydown', onKey)
  window.addEventListener('scroll', onScroll, true)

  return {
    // Build the payload the backend's BehaviorPayload schema expects.
    getPayload(honeypotValue = '') {
      return {
        form_load_time: loadTime,
        submit_time: Date.now(),
        mouse_events: mouse.slice(),
        key_intervals_ms: keyIntervals.slice(),
        scroll_events: scrolls.slice(),
        honeypot: honeypotValue,
      }
    },
    // Detach listeners (call on unmount to avoid leaks).
    stop() {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', onScroll, true)
    },
  }
}
