/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          0: '#0f172a',
          1: '#1e293b',
          2: '#334155',
        },
      },
      typography: {
        DEFAULT: {
          css: {
            color: '#e2e8f0',
            'h1,h2,h3,h4': { color: '#f1f5f9' },
            code: { color: '#a5f3fc', background: '#1e293b', borderRadius: '4px', padding: '2px 4px' },
            'pre code': { background: 'transparent', padding: 0 },
            pre: { background: '#1e293b', borderRadius: '8px' },
            a: { color: '#60a5fa' },
            strong: { color: '#f1f5f9' },
          },
        },
      },
    },
  },
  plugins: [],
}
