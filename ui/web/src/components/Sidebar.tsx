import { MessageSquare, Plus, Trash2, FileText, Clock } from 'lucide-react'
import type { Session } from '../types'

interface Props {
  sessions: Session[]
  activeId: string | null
  loading: boolean
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  onShowDocs: () => void
}

function formatTime(ts: number | null): string {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const now = new Date()
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86400000)
  if (diffDays === 0) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return d.toLocaleDateString([], { weekday: 'short' })
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function Sidebar({ sessions, activeId, loading, onSelect, onNew, onDelete, onShowDocs }: Props) {
  return (
    <aside className="w-64 flex flex-col bg-slate-900 border-r border-slate-800 shrink-0">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
            <MessageSquare size={14} className="text-white" />
          </div>
          <span className="font-semibold text-slate-100 text-sm">Personal AI</span>
        </div>
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors"
        >
          <Plus size={15} />
          New chat
        </button>
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto py-2">
        {loading ? (
          <div className="px-4 py-3 text-slate-500 text-sm">Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="px-4 py-3 text-slate-500 text-sm">No chats yet</div>
        ) : (
          sessions.map(s => (
            <div
              key={s.session_id}
              className={`group flex items-start gap-2 px-3 py-2.5 mx-1 rounded-lg cursor-pointer transition-colors ${
                activeId === s.session_id
                  ? 'bg-slate-700 text-slate-100'
                  : 'hover:bg-slate-800 text-slate-300'
              }`}
              onClick={() => onSelect(s.session_id)}
            >
              <MessageSquare size={14} className="mt-0.5 shrink-0 text-slate-500" />
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate leading-tight">
                  {s.title || 'New chat'}
                </div>
                {s.last_message && (
                  <div className="text-xs text-slate-500 truncate mt-0.5 leading-tight">
                    {s.last_message}
                  </div>
                )}
                <div className="flex items-center gap-1 mt-0.5 text-slate-600 text-xs">
                  <Clock size={10} />
                  {formatTime(s.last_activity)}
                </div>
              </div>
              <button
                onClick={e => { e.stopPropagation(); onDelete(s.session_id) }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-slate-600 text-slate-400 hover:text-red-400 transition-all shrink-0"
                title="Delete chat"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-slate-800">
        <button
          onClick={onShowDocs}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-sm transition-colors"
        >
          <FileText size={15} />
          Documents
        </button>
      </div>
    </aside>
  )
}
