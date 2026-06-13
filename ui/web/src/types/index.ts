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
  metrics?: {
    ttft?: number
    tgs?: number
    route?: string
    model?: string
    tokens?: number
    vram_bytes?: number
    ram_bytes?: number
  }
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

export interface IngestionStatus {
  doc_id: string
  status: string        // = stage (kept for back-compat)
  stage: string
  progress: number      // 0–100
  detail: string
  error: string
  failed_stage?: string
}

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'sources'; sources: Source[] }
  | { type: 'done'; ttft?: number; tgs?: number; route?: string; model?: string; tokens?: number; vram_bytes?: number; ram_bytes?: number }
  | { type: 'error'; message: string }

export type BenchmarkSSEEvent =
  | { type: 'log'; text: string }
  | { type: 'done'; error: string; cancelled: boolean; results_dir: string; completed_models: number; total_models: number }

export interface BenchmarkStatus {
  running: boolean
  completed: boolean
  cancelled: boolean
  current_suite: string
  current_model: string
  current_step: number
  current_total: number
  current_label: string
  completed_models: number
  total_models: number
  error: string
  results_dir: string
  elapsed_s: number
}

export interface BenchmarkModel {
  id: string
  name: string
  family: string
  params_b: number | null
  quant: string
}

export interface SuiteColumn {
  key: string
  label: string
}

export interface SuiteTable {
  label: string
  columns: SuiteColumn[]
  rows: Record<string, string | number>[]
}

export interface BenchmarkSummary {
  run_id: string
  suites: Record<string, SuiteTable>
}

export interface BenchmarkRunItem {
  run_id: string
  timestamp: number
}

export interface ModelsList {
  models: string[]
  current_model: string
}

export type ModelSwitchEvent = 
  | { status: 'unloading'; model: string }
  | { status: 'loading'; model: string }
  | { status: 'ready'; model: string }
  | { status: 'error'; message: string }
