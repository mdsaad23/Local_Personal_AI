export interface Session {
  session_id: string
  title: string
  started_at: number
  turn_count: number
  last_message: string | null
  last_activity: number | null
}

export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  has_image: boolean
  timestamp: number
  streaming?: boolean
  sources?: Source[]
  metrics?: { ttft?: number; tgs?: number; route?: string; model?: string }
}

export interface Source {
  source: string
  page: string | number
  section: string
  score: number | null
}

export interface Document {
  doc_id: string
  source: string
  file_type: string
  chunk_count: number
}

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'sources'; sources: Source[] }
  | { type: 'done'; ttft?: number; tgs?: number; route?: string; model?: string }
  | { type: 'error'; message: string }

export type BenchmarkSSEEvent =
  | { type: 'log'; text: string }
  | { type: 'done'; error: string; results_dir: string; completed_models: number; total_models: number }

export interface BenchmarkStatus {
  running: boolean
  completed: boolean
  current_suite: string
  current_model: string
  completed_models: number
  total_models: number
  error: string
  results_dir: string
  elapsed_s: number
}
