/** @type {import('tailwindcss').Config} */
// Verum's design tokens live here. Every color/type decision in the UI is
// derived from this file, so the look stays consistent and intentional.
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0B1220',      // primary text — deep navy-ink
        paper: '#F5F7FA',    // app background — cool paper
        panel: '#FFFFFF',    // card surfaces
        line: '#E4E8EF',     // hairline borders
        muted: '#5B6472',    // secondary text
        brand: '#0E4F5C',    // deep teal — the "forensic trust" accent
        verify: '#0F766E',   // teal   — low concern / authentic
        review: '#B45309',   // amber  — medium concern / needs review
        alert: '#BE123C',    // crimson — high concern (used sparingly)
        highlight: '#FEF3C7' // evidence marker background
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        panel: '0 1px 2px rgba(11,18,32,0.04), 0 10px 30px -18px rgba(11,18,32,0.25)',
      },
      borderRadius: { xl2: '1.1rem' },
    },
  },
  plugins: [],
}
