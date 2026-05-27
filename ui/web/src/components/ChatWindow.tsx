import { useEffect, useRef, useCallback } from 'react'
import { MessageSquare } from 'lucide-react'
import { api } from '../api/client'
import type { Message, Source } from '../types'
import MessageBubble from './MessageBubble'
import MessageInput from './MessageInput'

interface Props {
  sessionId: string | null
  messages: Message[]
  onNewChat: () => void
  onAppend: (msg: Message) => void
  onUpdateLast: (patch: Partial<Message>) => void
  onStreamComplete: () => void
}

const EMPTY_PLACEHOLDER = `Ask me anything. I'll search your documents, recall past context, and reason over what I know.`

export default function ChatWindow({
  sessionId, messages, onNewChat, onAppend, onUpdateLast, onStreamComplete
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const streamingRef = useRef(false)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = useCallback(async (text: string, image: File | null) => {
    if (streamingRef.current) return

    let sid = sessionId
    if (!sid) {
      // Auto-create a session if none selected
      try {
        const session = await api.createChat()
        sid = session.session_id
        // Notify parent to update state — but we continue with local sid immediately
        onNewChat()
      } catch {
        return
      }
    }

    const userMsg: Message = {
      role: 'user',
      content: text,
      has_image: !!image,
      timestamp: Date.now() / 1000,
    }
    onAppend(userMsg)

    const assistantMsg: Message = {
      role: 'assistant',
      content: '',
      has_image: false,
      timestamp: Date.now() / 1000,
      streaming: true,
    }
    onAppend(assistantMsg)
    streamingRef.current = true

    let accumulated = ''
    let sources: Source[] = []

    try {
      for await (const event of api.streamMessage(sid, text, image, s => { sources = s })) {
        if (event.type === 'token') {
          accumulated += event.content
          onUpdateLast({ content: accumulated, streaming: true, sources })
        } else if (event.type === 'sources') {
          sources = event.sources
          onUpdateLast({ sources })
        } else if (event.type === 'done') {
          onUpdateLast({
            content: accumulated,
            streaming: false,
            sources,
            metrics: {
              ttft: event.ttft,
              tgs: event.tgs,
              route: event.route,
              model: event.model,
            },
          })
        } else if (event.type === 'error') {
          onUpdateLast({ content: `Error: ${event.message}`, streaming: false })
        }
      }
    } catch (err) {
      onUpdateLast({ content: `Connection error. Is the server running?`, streaming: false })
    } finally {
      streamingRef.current = false
      onStreamComplete()
    }
  }, [sessionId, onNewChat, onAppend, onUpdateLast, onStreamComplete])

  const isStreaming = messages.some(m => m.streaming)

  return (
    <div className="flex flex-col h-full">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
              <MessageSquare size={24} className="text-indigo-400" />
            </div>
            <div>
              <h2 className="text-slate-200 font-semibold text-lg mb-1">Personal AI Assistant</h2>
              <p className="text-slate-500 text-sm max-w-xs">{EMPTY_PLACEHOLDER}</p>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="max-w-3xl w-full mx-auto self-stretch">
        <MessageInput disabled={isStreaming} onSend={handleSend} />
      </div>
    </div>
  )
}
