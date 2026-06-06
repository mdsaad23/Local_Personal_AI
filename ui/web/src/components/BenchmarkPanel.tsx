import { useState, useEffect, useRef, useCallback } from 'react'
import { X, Play, Square, CheckCircle, AlertCircle, Loader, ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '../api/client'
import type { BenchmarkStatus } from '../types'

interface Props {
  onClose: () => void
}

const SUITES = [
  { id: 'tools',  label: 'Tool Calling',   desc: '5 function-call tasks per model (~5 min)' },
  { id: 'niah',   label: 'Long Context',   desc: 'Needle in haystack 1k–32k tokens (~45 min)' },
  { id: 'coding', label: 'Coding',         desc: 'HumanEval+ pass@1, 20 problems per model (~3 hrs)' },
]

export default function BenchmarkPanel({ onClose }: Props) {
  const [selectedSuites, setSelectedSuites] = useState<string[]>(['tools', 'niah'])
  const [status, setStatus] = useState<BenchmarkStatus | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [streaming, setStreaming] = useState(false)
  const [done, setDone] = useState(false)
  const [logsOpen, setLogsOpen] = useState(true)
  const logsRef = useRef<HTMLDivElement>(null)

  // Poll status when running
  useEffect(() => {
    if (!streaming) return
    const interval = setInterval(async () => {
      try {
        const s = await api.getBenchmarkStatus()
        setStatus(s)
      } catch { /* server busy */ }
    }, 2000)
    return () => clearInterval(interval)
  }, [streaming])

  // Auto-scroll logs
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs])

  const startBenchmark = useCallback(async () => {
    if (selectedSuites.length === 0) return
    setLogs([])
    setDone(false)
    setStreaming(true)

    try {
      const result = await api.startBenchmark(selectedSuites)
      if (result.status === 'already_running') {
        setLogs(['Benchmark already running — reconnecting to stream…'])
      }
    } catch (e) {
      setLogs([`Failed to start: ${e}`])
      setStreaming(false)
      return
    }

    // Stream logs
    ;(async () => {
      try {
        for await (const event of api.streamBenchmark()) {
          if (event.type === 'log') {
            setLogs(prev => [...prev, event.text])
          } else if (event.type === 'done') {
            setDone(true)
            setStreaming(false)
            if (event.error) {
              setLogs(prev => [...prev, `ERROR: ${event.error}`])
            } else {
              setLogs(prev => [...prev, `✓ Complete — ${event.completed_models} model/suite pairs finished`])
            }
            const s = await api.getBenchmarkStatus()
            setStatus(s)
          }
        }
      } catch {
        setStreaming(false)
      }
    })()
  }, [selectedSuites])

  const stopBenchmark = useCallback(async () => {
    await api.stopBenchmark()
    setLogs(prev => [...prev, 'Stop requested — finishing current model…'])
  }, [])

  const toggleSuite = (id: string) => {
    setSelectedSuites(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    )
  }

  const progress = status && status.total_models > 0
    ? Math.round((status.completed_models / status.total_models) * 100)
    : 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <div>
          <h2 className="font-semibold text-slate-100">Benchmark — Test All Models</h2>
          <p className="text-xs text-slate-500 mt-0.5">Runs all 9 models sequentially. Results saved to data/benchmarks/</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <X size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">

        {/* Suite selection */}
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Select suites</h3>
          <div className="space-y-2">
            {SUITES.map(suite => (
              <label
                key={suite.id}
                className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedSuites.includes(suite.id)
                    ? 'bg-indigo-900/30 border-indigo-600/50'
                    : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedSuites.includes(suite.id)}
                  onChange={() => toggleSuite(suite.id)}
                  disabled={streaming}
                  className="mt-0.5 accent-indigo-500"
                />
                <div>
                  <div className="text-sm font-medium text-slate-200">{suite.label}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{suite.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="flex gap-3">
          {!streaming ? (
            <button
              onClick={startBenchmark}
              disabled={selectedSuites.length === 0}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play size={15} />
              {done ? 'Run again' : 'Test all models'}
            </button>
          ) : (
            <button
              onClick={stopBenchmark}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-red-700 hover:bg-red-600 text-white text-sm font-medium transition-colors"
            >
              <Square size={15} />
              Stop
            </button>
          )}
        </div>

        {/* Progress */}
        {(streaming || done) && status && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">
                {streaming ? (
                  <span className="flex items-center gap-1.5">
                    <Loader size={13} className="animate-spin" />
                    {status.current_suite && `${status.current_suite} › `}
                    {status.current_model || 'Starting…'}
                  </span>
                ) : done && !status.error ? (
                  <span className="flex items-center gap-1.5 text-green-400">
                    <CheckCircle size={13} /> Complete
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 text-red-400">
                    <AlertCircle size={13} /> {status.error || 'Stopped'}
                  </span>
                )}
              </span>
              <span className="text-slate-500 text-xs">
                {status.completed_models}/{status.total_models} · {status.elapsed_s}s
              </span>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Log output */}
        {logs.length > 0 && (
          <div className="border border-slate-700/50 rounded-lg overflow-hidden">
            <button
              onClick={() => setLogsOpen(o => !o)}
              className="w-full flex items-center justify-between px-4 py-2.5 bg-slate-800/80 text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              <span className="font-mono font-medium">Output log ({logs.length} lines)</span>
              {logsOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
            {logsOpen && (
              <div
                ref={logsRef}
                className="bg-slate-950 p-4 h-64 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5"
              >
                {logs.map((line, i) => (
                  <div key={i} className={`leading-relaxed ${
                    line.startsWith('ERROR') || line.startsWith('FATAL') ? 'text-red-400' :
                    line.startsWith('━') ? 'text-indigo-400 font-semibold mt-2' :
                    line.startsWith('✓') ? 'text-green-400' :
                    line.startsWith('  ') ? 'text-slate-400' : 'text-slate-300'
                  }`}>
                    {line}
                  </div>
                ))}
                {streaming && <div className="text-slate-600 animate-pulse">▊</div>}
              </div>
            )}
          </div>
        )}

        {/* Results note */}
        {done && status?.results_dir && (
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-sm text-slate-400">
            Results saved to:
            <code className="block mt-1 text-xs text-indigo-300 break-all">{status.results_dir}</code>
          </div>
        )}
      </div>
    </div>
  )
}
