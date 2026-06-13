import type {
  Session, Message, Document, SSEEvent, Source, BenchmarkSSEEvent,
  BenchmarkModel, BenchmarkSummary, IngestionStatus, ModelsList, ModelSwitchEvent, BenchmarkRunItem
} from '../types'

const BASE = '/api'

export const api = {
  // ── Chats ──────────────────────────────────────────────────────────────────
  listChats: (): Promise<Session[]> =>
    fetch(`${BASE}/chats`).then(r => r.json()),

  createChat: (): Promise<Session> =>
    fetch(`${BASE}/chats`, { method: 'POST' }).then(r => r.json()),

  deleteChat: (id: string): Promise<void> =>
    fetch(`${BASE}/chats/${id}`, { method: 'DELETE' }).then(() => {}),

  getMessages: (id: string): Promise<Message[]> =>
    fetch(`${BASE}/chats/${id}/messages`).then(r => r.json()),

  // ── Streaming ──────────────────────────────────────────────────────────────
  async *streamMessage(
    sessionId: string,
    message: string,
    image: File | null,
    onSources: (s: Source[]) => void,
  ): AsyncGenerator<SSEEvent> {
    const form = new FormData()
    form.append('message', message)
    if (image) form.append('image', image)

    const resp = await fetch(`${BASE}/chats/${sessionId}/stream`, {
      method: 'POST',
      body: form,
    })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

    const reader = resp.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const event = JSON.parse(line.slice(6)) as SSEEvent
          if (event.type === 'sources') onSources(event.sources)
          yield event
        } catch { /* ignore malformed */ }
      }
    }
  },

  // ── Documents ──────────────────────────────────────────────────────────────
  listDocuments: (): Promise<Document[]> =>
    fetch(`${BASE}/documents`).then(r => r.json()),

  uploadDocument: (file: File): Promise<{ source: string; chunks: number }> => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`${BASE}/documents`, { method: 'POST', body: form }).then(r => r.json())
  },

  // XHR-based upload so we can report byte-level upload progress (fetch can't).
  uploadDocumentProgress: (
    file: File,
    onProgress: (pct: number) => void,
  ): Promise<{ doc_id: string; source: string; status: string }> =>
    new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/documents`)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)) }
          catch { reject(new Error('Bad server response')) }
        } else {
          reject(new Error(`Upload failed: HTTP ${xhr.status}`))
        }
      }
      xhr.onerror = () => reject(new Error('Network error during upload'))
      xhr.send(form)
    }),

  getIngestionStatus: (docId: string): Promise<IngestionStatus> =>
    fetch(`${BASE}/documents/status/${encodeURIComponent(docId)}`).then(r => r.json()),

  deleteDocument: (docId: string): Promise<void> =>
    fetch(`${BASE}/documents/${docId}`, { method: 'DELETE' }).then(() => {}),

  reingestDocument: (docId: string): Promise<void> =>
    fetch(`${BASE}/documents/${docId}/reingest`, { method: 'POST' }).then(() => {}),

  // ── System ─────────────────────────────────────────────────────────────────
  health: () => fetch(`${BASE}/health`).then(r => r.json()),

  // ── Benchmark ──────────────────────────────────────────────────────────────
  getBenchmarkModels: (): Promise<BenchmarkModel[]> =>
    fetch(`${BASE}/benchmark/models`).then(r => r.json()),

  startBenchmark: (suites: string[], models: string[]): Promise<{ status: string; message?: string }> =>
    fetch(`${BASE}/benchmark/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ suites, models }),
    }).then(r => r.json()),

  stopBenchmark: (): Promise<{ status: string }> =>
    fetch(`${BASE}/benchmark/stop`, { method: 'POST' }).then(r => r.json()),

  getBenchmarkStatus: () =>
    fetch(`${BASE}/benchmark/status`).then(r => r.json()),

  getBenchmarkSummary: (): Promise<BenchmarkSummary> =>
    fetch(`${BASE}/benchmark/summary`).then(r => r.json()),

  getBenchmarkHistory: (): Promise<BenchmarkRunItem[]> =>
    fetch(`${BASE}/benchmark/history`).then(r => r.json()),

  getBenchmarkSummaryById: (runId: string): Promise<BenchmarkSummary> =>
    fetch(`${BASE}/benchmark/summary/${runId}`).then(r => r.json()),

  async *streamBenchmark(): AsyncGenerator<BenchmarkSSEEvent> {
    const resp = await fetch(`${BASE}/benchmark/stream`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const reader = resp.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          yield JSON.parse(line.slice(6)) as BenchmarkSSEEvent
        } catch { /* ignore malformed */ }
      }
    }
  },

  // ── Models ─────────────────────────────────────────────────────────────────
  getModels: (): Promise<ModelsList> =>
    fetch(`${BASE}/models`).then(r => r.json()),

  async *switchModel(modelId: string): AsyncGenerator<ModelSwitchEvent> {
    const resp = await fetch(`${BASE}/models/active`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    
    const reader = resp.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          yield JSON.parse(line.slice(6)) as ModelSwitchEvent
        } catch { /* ignore malformed */ }
      }
    }
  },
}
