import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronDown, ChevronUp, BookOpen } from 'lucide-react'
import { useState } from 'react'
import type { Message } from '../types'

interface Props {
  message: Message
}

export default function MessageBubble({ message }: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const isUser = message.role === 'user'
  const formatGB = (bytes: number) => (bytes / 1024 / 1024 / 1024).toFixed(1) + ' GB'

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} group`}>
      {/* Avatar */}
      <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 mt-1 ${
        isUser ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-300'
      }`}>
        {isUser ? 'U' : 'AI'}
      </div>

      {/* Bubble */}
      <div className={`max-w-[75%] flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Image preview */}
        {message.has_image && (
          <div className="text-xs text-slate-500 italic mb-1">📎 Image attached</div>
        )}

        <div className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-indigo-600 text-white rounded-tr-sm'
            : 'bg-slate-800 text-slate-200 rounded-tl-sm'
        }`}>
          {isUser ? (
            <span className="whitespace-pre-wrap">{message.content}</span>
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {message.streaming && <span className="cursor" />}
            </div>
          )}
        </div>

        {/* Metrics row */}
        {message.metrics && (
          <div className="flex gap-3 text-xs text-slate-600 px-1">
            {message.metrics.model && (
              <span className="text-slate-500">{message.metrics.model}</span>
            )}
            {message.metrics.route && (
              <span className="bg-slate-800 px-1.5 py-0.5 rounded text-slate-400">
                {message.metrics.route}
              </span>
            )}
            {message.metrics.ttft != null && (
              <span>TTFT {message.metrics.ttft.toFixed(2)}s</span>
            )}
            {message.metrics.tokens != null && (
              <span>{message.metrics.tokens} tokens</span>
            )}
            {message.metrics.tgs != null && (
              <span>{message.metrics.tgs.toFixed(1)} t/s</span>
            )}
            {message.metrics.vram_bytes != null && (
              <span className="text-emerald-500/70">VRAM {formatGB(message.metrics.vram_bytes)}</span>
            )}
            {message.metrics.ram_bytes != null && message.metrics.ram_bytes > 0 && (
              <span className="text-amber-500/70">RAM {formatGB(message.metrics.ram_bytes)}</span>
            )}
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setSourcesOpen(o => !o)}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors px-1"
            >
              <BookOpen size={11} />
              {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
              {sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>
            {sourcesOpen && (
              <div className="mt-1 space-y-1">
                {message.sources.map((src, i) => (
                  <div key={i} className="bg-slate-800/60 rounded-lg px-3 py-2 text-xs border border-slate-700/50">
                    <div className="font-medium text-slate-300 truncate">{src.source}</div>
                    <div className="text-slate-500 mt-0.5">
                      {src.section && <span className="mr-2">{src.section}</span>}
                      {src.page && <span>p. {src.page}</span>}
                      {src.score != null && (
                        <span className="ml-2 text-slate-600">score: {src.score.toFixed(3)}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
