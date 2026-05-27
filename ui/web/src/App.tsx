import { useState, useEffect, useCallback } from 'react'
import { api } from './api/client'
import type { Session, Message } from './types'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import DocumentPanel from './components/DocumentPanel'

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [showDocs, setShowDocs] = useState(false)
  const [loadingChats, setLoadingChats] = useState(true)

  const refreshSessions = useCallback(async () => {
    try {
      const list = await api.listChats()
      setSessions(list)
    } catch {
      // server may not be up yet
    }
  }, [])

  useEffect(() => {
    refreshSessions().finally(() => setLoadingChats(false))
  }, [refreshSessions])

  const selectChat = useCallback(async (id: string) => {
    setActiveId(id)
    setShowDocs(false)
    try {
      const msgs = await api.getMessages(id)
      setMessages(msgs)
    } catch {
      setMessages([])
    }
  }, [])

  const newChat = useCallback(async () => {
    const session = await api.createChat()
    setSessions(prev => [session, ...prev])
    setActiveId(session.session_id)
    setMessages([])
    setShowDocs(false)
  }, [])

  const deleteChat = useCallback(async (id: string) => {
    await api.deleteChat(id)
    setSessions(prev => prev.filter(s => s.session_id !== id))
    if (activeId === id) {
      setActiveId(null)
      setMessages([])
    }
  }, [activeId])

  const appendMessage = useCallback((msg: Message) => {
    setMessages(prev => [...prev, msg])
  }, [])

  const updateLastAssistant = useCallback((patch: Partial<Message>) => {
    setMessages(prev => {
      const next = [...prev]
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === 'assistant') {
          next[i] = { ...next[i], ...patch }
          break
        }
      }
      return next
    })
  }, [])

  const onStreamComplete = useCallback(() => {
    refreshSessions()
  }, [refreshSessions])

  return (
    <div className="flex h-screen bg-slate-900 text-slate-200 overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        loading={loadingChats}
        onSelect={selectChat}
        onNew={newChat}
        onDelete={deleteChat}
        onShowDocs={() => { setShowDocs(true); setActiveId(null) }}
      />

      <main className="flex-1 flex flex-col min-w-0">
        {showDocs ? (
          <DocumentPanel onClose={() => setShowDocs(false)} />
        ) : (
          <ChatWindow
            sessionId={activeId}
            messages={messages}
            onNewChat={newChat}
            onAppend={appendMessage}
            onUpdateLast={updateLastAssistant}
            onStreamComplete={onStreamComplete}
          />
        )}
      </main>
    </div>
  )
}
