import type { Session, Message, Document, SSEEvent, Source, BenchmarkSSEEvent } from '../types'

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

  deleteDocument: (docId: string): Promise<void> =>
    fetch(`${BASE}/documents/${docId}`, { method: 'DELETE' }).then(() => {}),

  reingestDocument: (docId: string): Promise<void> =>
    fetch(`${BASE}/documents/${docId}/reingest`, { method: 'POST' }).then(() => {}),

  // ── System ─────────────────────────────────────────────────────────────────
  health: () => fetch(`${BASE}/health`).then(r => r.json()),

  // ── Benchmark ──────────────────────────────────────────────────────────────
  startBenchmark: (suites: string[]): Promise<{ status: string; message?: string }> =>
    fetch(`${BASE}/benchmark/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(suites),
    }).then(r => r.json()),

  stopBenchmark: (): Promise<{ status: string }> =>
    fetch(`${BASE}/benchmark/stop`, { method: 'POST' }).then(r => r.json()),

  getBenchmarkStatus: () =>
    fetch(`${BASE}/benchmark/status`).then(r => r.json()),

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
}
