import { useState, useEffect, useCallback, useRef } from 'react'
import { Upload, Trash2, X, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import { api } from '../api/client'
import type { Document } from '../types'

interface Props {
  onClose: () => void
}

type UploadState =
  | { status: 'idle' }
  | { status: 'uploading'; name: string }
  | { status: 'ingesting'; name: string; stage: string }
  | { status: 'done'; name: string }
  | { status: 'error'; message: string }

const STAGE_LABELS: Record<string, string> = {
  queued:         'Queued…',
  parsing:        'Parsing document…',
  chunking:       'Splitting into chunks…',
  embedding:      'Generating embeddings…',
  storing:        'Storing in vector DB…',
  building_graph: 'Building knowledge graph…',
  complete:       'Done',
}

export default function DocumentPanel({ onClose }: Props) {
  const [docs, setDocs] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [uploadState, setUploadState] = useState<UploadState>({ status: 'idle' })
  const [deleting, setDeleting] = useState<Set<string>>(new Set())
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const refresh = useCallback(async () => {
    try {
      const list = await api.listDocuments()
      setDocs(list)
    } catch { /* server may not be up yet */ }
  }, [])

  useEffect(() => {
    refresh().finally(() => setLoading(false))
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [refresh])

  const pollStatus = useCallback((docId: string, name: string) => {
    if (pollRef.current) clearInterval(pollRef.current)

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/documents/status/${encodeURIComponent(docId)}`)
        const data = await res.json()
        const stage: string = data.status || 'unknown'

        if (stage === 'complete') {
          clearInterval(pollRef.current!)
          setUploadState({ status: 'done', name })
          await refresh()
          setTimeout(() => setUploadState({ status: 'idle' }), 3000)
        } else if (stage.startsWith('error')) {
          clearInterval(pollRef.current!)
          setUploadState({ status: 'error', message: stage })
        } else {
          setUploadState({ status: 'ingesting', name, stage })
        }
      } catch { /* server busy, retry */ }
    }, 1500)
  }, [refresh])

  const handleUpload = useCallback(async (file: File) => {
    setUploadState({ status: 'uploading', name: file.name })
    try {
      const result = await api.uploadDocument(file)
      // Server returns immediately with status='ingesting'
      const docId = (result as any).doc_id || file.name
      setUploadState({ status: 'ingesting', name: file.name, stage: 'queued' })
      pollStatus(docId, file.name)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload failed'
      setUploadState({ status: 'error', message: msg })
    }
  }, [pollStatus])

  const handleDelete = useCallback(async (docId: string) => {
    setDeleting(prev => new Set(prev).add(docId))
    try {
      await api.deleteDocument(docId)
      setDocs(prev => prev.filter(d => d.doc_id !== docId))
    } finally {
      setDeleting(prev => { const s = new Set(prev); s.delete(docId); return s })
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }, [handleUpload])

  const FILE_ICONS: Record<string, string> = { pdf: '📄', docx: '📝', md: '📋', txt: '📃' }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <h2 className="font-semibold text-slate-100">Documents</h2>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <X size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Drop zone */}
        <div
          className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-8 text-center transition-colors cursor-pointer"
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            className="hidden"
            accept=".pdf,.docx,.md,.txt,.png,.jpg,.jpeg"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f) }}
          />
          <Upload size={28} className="text-slate-500 mx-auto mb-2" />
          <p className="text-slate-400 text-sm font-medium">Drop a file or click to upload</p>
          <p className="text-slate-600 text-xs mt-1">PDF · DOCX · MD · TXT · Images</p>
        </div>

        {/* Upload / ingestion status */}
        {uploadState.status !== 'idle' && (
          <div className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm border ${
            uploadState.status === 'error'
              ? 'bg-red-900/30 border-red-700/50 text-red-300'
              : uploadState.status === 'done'
              ? 'bg-green-900/30 border-green-700/50 text-green-300'
              : 'bg-slate-800 border-slate-700 text-slate-300'
          }`}>
            {(uploadState.status === 'uploading' || uploadState.status === 'ingesting') &&
              <Loader size={16} className="animate-spin shrink-0" />}
            {uploadState.status === 'done' && <CheckCircle size={16} className="shrink-0" />}
            {uploadState.status === 'error' && <AlertCircle size={16} className="shrink-0" />}
            <div className="min-w-0">
              <div className="truncate">
                {uploadState.status === 'uploading' && `Uploading ${uploadState.name}…`}
                {uploadState.status === 'ingesting' && (
                  <span>
                    <span className="text-slate-400">{uploadState.name} — </span>
                    {STAGE_LABELS[uploadState.stage] ?? uploadState.stage}
                  </span>
                )}
                {uploadState.status === 'done' && `${uploadState.name} — indexed successfully`}
                {uploadState.status === 'error' && uploadState.message}
              </div>
              {uploadState.status === 'ingesting' && (
                <div className="text-xs text-slate-500 mt-0.5">
                  This takes 2–5 min (embeddings + knowledge graph)
                </div>
              )}
            </div>
          </div>
        )}

        {/* Document list */}
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Indexed documents ({docs.length})
          </h3>
          {loading ? (
            <div className="text-slate-500 text-sm">Loading…</div>
          ) : docs.length === 0 ? (
            <div className="text-slate-600 text-sm">No documents yet. Upload one above.</div>
          ) : (
            <div className="space-y-2">
              {docs.map(doc => (
                <div
                  key={doc.doc_id}
                  className="flex items-center gap-3 px-4 py-3 bg-slate-800 rounded-lg border border-slate-700/50 group"
                >
                  <span className="text-lg shrink-0">{FILE_ICONS[doc.file_type] ?? '📎'}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-slate-200 truncate font-medium">{doc.source}</div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      {doc.chunk_count} chunks · {doc.file_type.toUpperCase()}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.doc_id)}
                    disabled={deleting.has(doc.doc_id)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-red-400 transition-all disabled:opacity-40"
                    title="Remove from index"
                  >
                    {deleting.has(doc.doc_id)
                      ? <Loader size={14} className="animate-spin" />
                      : <Trash2 size={14} />}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
