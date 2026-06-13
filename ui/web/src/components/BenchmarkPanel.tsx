import { useState, useEffect, useRef, useCallback } from 'react'
import { X, Play, Square, CheckCircle, AlertCircle, Loader, ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '../api/client'
import type { BenchmarkStatus, BenchmarkModel, BenchmarkSummary, BenchmarkRunItem } from '../types'

interface Props {
  onClose: () => void
}

const SUITES = [
  { id: 'tools',  label: 'Tool Calling',      desc: '5 function-call tasks per model (~5 min)' },
  { id: 'niah',   label: 'Long Context',       desc: 'Needle in haystack 1k–32k tokens (~45 min)' },
  { id: 'needle', label: 'Positional Recall',  desc: 'codeneedle — reproduce functions from long context (~10 min)' },
  { id: 'coding', label: 'Coding',             desc: 'HumanEval+ pass@1, 20 problems per model (~3 hrs)' },
]

export default function BenchmarkPanel({ onClose }: Props) {
  const [models, setModels] = useState<BenchmarkModel[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [selectedSuites, setSelectedSuites] = useState<string[]>(['tools', 'needle'])
  const [status, setStatus] = useState<BenchmarkStatus | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [streaming, setStreaming] = useState(false)
  const [done, setDone] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [cancelled, setCancelled] = useState(false)
  const [logsOpen, setLogsOpen] = useState(true)
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null)
  const [history, setHistory] = useState<BenchmarkRunItem[]>([])
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>('latest')
  const logsRef = useRef<HTMLDivElement>(null)

  // Load selectable models — default to all selected
  useEffect(() => {
    api.getBenchmarkModels()
      .then(m => { setModels(m); setSelectedModels(m.map(x => x.id)) })
      .catch(() => setModels([]))
      
    // Load the most recent summary if available
    api.getBenchmarkSummary()
      .then(s => setSummary(s))
      .catch(() => setSummary(null))
      
    // Load history
    api.getBenchmarkHistory()
      .then(h => setHistory(h))
      .catch(() => setHistory([]))
  }, [])
  
  // Fetch specific summary when history selection changes
  useEffect(() => {
    if (selectedHistoryId === 'latest') {
      if (!streaming) {
        api.getBenchmarkSummary()
          .then(s => setSummary(s))
          .catch(() => setSummary(null))
      }
    } else {
      api.getBenchmarkSummaryById(selectedHistoryId)
        .then(s => setSummary(s))
        .catch(() => setSummary(null))
    }
  }, [selectedHistoryId, streaming])

  // Poll status when running
  useEffect(() => {
    if (!streaming) return
    const interval = setInterval(async () => {
      try { setStatus(await api.getBenchmarkStatus()) } catch { /* server busy */ }
    }, 2000)
    return () => clearInterval(interval)
  }, [streaming])

  // Auto-scroll logs
  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight
  }, [logs])

  const startBenchmark = useCallback(async () => {
    if (selectedSuites.length === 0 || selectedModels.length === 0) return
    setLogs([]); setDone(false); setCancelled(false); setCancelling(false)
    setSummary(null); setStreaming(true)

    try {
      const result = await api.startBenchmark(selectedSuites, selectedModels)
      if (result.status === 'already_running') {
        setLogs(['Benchmark already running — reconnecting to stream…'])
      }
    } catch (e) {
      setLogs([`Failed to start: ${e}`]); setStreaming(false); return
    }

    ;(async () => {
      try {
        for await (const event of api.streamBenchmark()) {
          if (event.type === 'log') {
            setLogs(prev => [...prev, event.text])
          } else if (event.type === 'done') {
            setDone(true); setStreaming(false); setCancelling(false)
            setCancelled(!!event.cancelled)
            if (event.error) setLogs(prev => [...prev, `ERROR: ${event.error}`])
            else if (event.cancelled) setLogs(prev => [...prev, `■ Cancelled — ${event.completed_models} model/suite pairs finished before stop`])
            else setLogs(prev => [...prev, `✓ Complete — ${event.completed_models} model/suite pairs finished`])
            try { setStatus(await api.getBenchmarkStatus()) } catch { /* ignore */ }
            try { setSummary(await api.getBenchmarkSummary()) } catch { /* ignore */ }
          }
        }
      } catch { setStreaming(false) }
    })()
  }, [selectedSuites, selectedModels])

  const stopBenchmark = useCallback(async () => {
    setCancelling(true)
    setLogs(prev => [...prev, 'Cancelling — stopping safely, keeping results so far…'])
    await api.stopBenchmark()
  }, [])

  const toggleSuite = (id: string) =>
    setSelectedSuites(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id])
  const toggleModel = (id: string) =>
    setSelectedModels(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id])
  const allModelsSelected = models.length > 0 && selectedModels.length === models.length

  const progress = status && status.total_models > 0
    ? Math.round((status.completed_models / status.total_models) * 100) : 0

  const canStart = selectedSuites.length > 0 && selectedModels.length > 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <div>
          <h2 className="font-semibold text-slate-100">Benchmark — Automated Model Testing</h2>
          <p className="text-xs text-slate-500 mt-0.5">Select models + suites. Each runs sequentially; results saved to data/benchmarks/</p>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors">
          <X size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">

        {/* Model selection */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Select models ({selectedModels.length}/{models.length})
            </h3>
            <button
              onClick={() => setSelectedModels(allModelsSelected ? [] : models.map(m => m.id))}
              disabled={streaming}
              className="text-xs text-indigo-400 hover:text-indigo-300 disabled:opacity-40"
            >
              {allModelsSelected ? 'Deselect all' : 'Select all'}
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {models.map(model => (
              <label
                key={model.id}
                className={`flex items-start gap-2.5 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                  selectedModels.includes(model.id)
                    ? 'bg-indigo-900/30 border-indigo-600/50'
                    : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.id)}
                  onChange={() => toggleModel(model.id)}
                  disabled={streaming}
                  className="mt-0.5 accent-indigo-500"
                />
                <div className="min-w-0">
                  <div className="text-sm font-medium text-slate-200 truncate">{model.name}</div>
                  <div className="text-xs text-slate-500 truncate">{model.id}</div>
                </div>
              </label>
            ))}
            {models.length === 0 && (
              <div className="col-span-2 text-xs text-slate-500">No models found in config/models.yaml</div>
            )}
          </div>
        </div>

        {/* Suite selection */}
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Select test suites</h3>
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
              disabled={!canStart}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play size={15} />
              {done ? 'Run again' : `Test ${selectedModels.length} model${selectedModels.length === 1 ? '' : 's'}`}
            </button>
          ) : (
            <button
              onClick={stopBenchmark}
              disabled={cancelling}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-red-700 hover:bg-red-600 text-white text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Square size={15} />
              {cancelling ? 'Cancelling…' : 'Cancel run'}
            </button>
          )}
        </div>

        {/* Progress */}
        {(streaming || done) && status && (
          <div className="space-y-4">
            {/* Overall progress */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">
                  {streaming ? (
                    <span className="flex items-center gap-1.5">
                      <Loader size={13} className="animate-spin" />
                      {cancelling ? 'Cancelling…' : `Overall — ${status.current_suite || 'starting'}`}
                    </span>
                  ) : cancelled ? (
                    <span className="flex items-center gap-1.5 text-amber-400"><Square size={12} /> Cancelled — partial results below</span>
                  ) : done && !status.error ? (
                    <span className="flex items-center gap-1.5 text-green-400"><CheckCircle size={13} /> Complete</span>
                  ) : (
                    <span className="flex items-center gap-1.5 text-red-400"><AlertCircle size={13} /> {status.error || 'Stopped'}</span>
                  )}
                </span>
                <span className="text-slate-500 text-xs">{status.completed_models}/{status.total_models} pairs · {status.elapsed_s}s</span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-indigo-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
              </div>
            </div>

            {/* Per-model progress (only while running) */}
            {streaming && status.current_model && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 truncate">{status.current_model}</span>
                  <span className="text-slate-500">
                    {status.current_total > 0
                      ? `${status.current_label || 'working'} · ${status.current_step}/${status.current_total}`
                      : (status.current_label || 'in progress…')}
                  </span>
                </div>
                <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                  {status.current_total > 0 ? (
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                      style={{ width: `${Math.round((status.current_step / status.current_total) * 100)}%` }}
                    />
                  ) : (
                    <div className="h-full w-1/3 bg-emerald-600/70 rounded-full animate-pulse" />
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results — one table section per suite */}
        {summary && Object.keys(summary.suites).length > 0 && (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Results</h3>
              {history.length > 0 && (
                <select
                  value={selectedHistoryId}
                  onChange={(e) => setSelectedHistoryId(e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-md px-2 py-1 outline-none cursor-pointer"
                >
                  <option value="latest">Latest / Current</option>
                  {history.map(run => (
                    <option key={run.run_id} value={run.run_id}>
                      {new Date(run.timestamp * 1000).toLocaleString()}
                    </option>
                  ))}
                </select>
              )}
            </div>
            
            {Object.entries(summary.suites).map(([suiteId, table]) => (
              <div key={suiteId} className="border border-slate-700/50 rounded-lg overflow-hidden">
                <div className="px-4 py-2.5 bg-slate-800/80 text-sm font-medium text-slate-200">{table.label}</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700/50 text-left">
                        {table.columns.map(col => (
                          <th key={col.key} className="px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider whitespace-nowrap">
                            {col.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {table.rows.map((row, ri) => (
                        <tr key={ri} className="border-b border-slate-800/50 last:border-0 hover:bg-slate-800/30">
                          {table.columns.map((col, ci) => (
                            <td key={col.key} className={`px-4 py-2 whitespace-nowrap ${ci === 0 ? 'text-slate-200 font-medium' : 'text-slate-400'}`}>
                              {row[col.key] ?? '—'}
                            </td>
                          ))}
                        </tr>
                      ))}
                      {table.rows.length === 0 && (
                        <tr><td colSpan={table.columns.length} className="px-4 py-3 text-xs text-slate-500">No results</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
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
              <div ref={logsRef} className="bg-slate-950 p-4 h-64 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
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
