import { useRef, useState, useCallback } from 'react'
import { Send, Paperclip, X, Image } from 'lucide-react'

interface Props {
  disabled: boolean
  onSend: (message: string, image: File | null) => void
}

export default function MessageInput({ disabled, onSend }: Props) {
  const [text, setText] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleImageSelect = useCallback((file: File) => {
    setImage(file)
    const url = URL.createObjectURL(file)
    setImagePreview(url)
  }, [])

  const clearImage = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview)
    setImage(null)
    setImagePreview(null)
    if (fileRef.current) fileRef.current.value = ''
  }, [imagePreview])

  const handleSubmit = useCallback(() => {
    const msg = text.trim()
    if (!msg && !image) return
    onSend(msg, image)
    setText('')
    clearImage()
    textareaRef.current?.focus()
  }, [text, image, onSend, clearImage])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const file = Array.from(e.clipboardData.files).find(f => f.type.startsWith('image/'))
    if (file) {
      e.preventDefault()
      handleImageSelect(file)
    }
  }, [handleImageSelect])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = Array.from(e.dataTransfer.files).find(f => f.type.startsWith('image/'))
    if (file) handleImageSelect(file)
  }, [handleImageSelect])

  return (
    <div
      className="border-t border-slate-800 bg-slate-900 p-4"
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
    >
      {/* Image preview strip */}
      {imagePreview && (
        <div className="flex items-center gap-2 mb-2 px-1">
          <div className="relative">
            <img src={imagePreview} alt="preview" className="h-16 w-16 object-cover rounded-lg border border-slate-700" />
            <button
              onClick={clearImage}
              className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-slate-600 hover:bg-red-500 rounded-full flex items-center justify-center transition-colors"
            >
              <X size={10} className="text-white" />
            </button>
          </div>
          <div className="text-xs text-slate-500">
            <Image size={12} className="inline mr-1" />
            {image?.name}
          </div>
        </div>
      )}

      <div className="flex items-end gap-2">
        {/* Image attach */}
        <button
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
          title="Attach image (or paste / drag-drop)"
          className="p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
        >
          <Paperclip size={18} />
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={e => {
            const f = e.target.files?.[0]
            if (f) handleImageSelect(f)
          }}
        />

        {/* Text area */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled}
          placeholder={disabled ? 'Thinking…' : 'Message (Shift+Enter for newline)'}
          rows={1}
          className="flex-1 resize-none bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed max-h-48 overflow-y-auto"
          style={{ minHeight: '42px' }}
          onInput={e => {
            const el = e.currentTarget
            el.style.height = 'auto'
            el.style.height = Math.min(el.scrollHeight, 192) + 'px'
          }}
        />

        {/* Send */}
        <button
          onClick={handleSubmit}
          disabled={disabled || (!text.trim() && !image)}
          className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
        >
          <Send size={16} />
        </button>
      </div>

      <div className="text-xs text-slate-600 mt-1.5 px-1">
        Enter to send · Shift+Enter for newline · Paste or drag images
      </div>
    </div>
  )
}
