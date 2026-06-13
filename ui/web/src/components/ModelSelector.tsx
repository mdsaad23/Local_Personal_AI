import React, { useEffect, useState } from 'react'
import { api } from '../api/client'

interface ModelSelectorProps {
  onSelectModel: (modelId: string) => void
  disabled?: boolean
}

export function ModelSelector({ onSelectModel, disabled }: ModelSelectorProps) {
  const [models, setModels] = useState<string[]>([])
  const [currentModel, setCurrentModel] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    let mounted = true
    api.getModels()
      .then(res => {
        if (!mounted) return
        setModels(res.models)
        setCurrentModel(res.current_model)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load models:', err)
        setLoading(false)
      })
    return () => { mounted = false }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value
    if (newModel && newModel !== currentModel) {
      setCurrentModel(newModel)
      onSelectModel(newModel)
    }
  }

  if (loading) {
    return <div className="model-selector-loading">Loading models...</div>
  }

  return (
    <div className="model-selector">
      <select 
        value={currentModel} 
        onChange={handleChange}
        disabled={disabled}
        className="model-dropdown"
      >
        {models.map(m => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    </div>
  )
}
