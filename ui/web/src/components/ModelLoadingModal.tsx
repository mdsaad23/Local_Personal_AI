import React from 'react'

interface ModelLoadingModalProps {
  status: string
}

export function ModelLoadingModal({ status }: ModelLoadingModalProps) {
  return (
    <div className="model-loading-overlay">
      <div className="model-loading-modal">
        <h3>Switching Model</h3>
        <div className="spinner"></div>
        <p className="status-text">{status}</p>
      </div>
    </div>
  )
}
