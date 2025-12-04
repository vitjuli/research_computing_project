'use client'

import { useState, useEffect } from 'react'
import { listModels, deleteModel, ModelInfo } from '@/lib/api'

interface ModelListProps {
  selectedModelId: string | null
  onSelectModel: (modelId: string) => void
  onRefresh: () => void
}

export default function ModelList({
  selectedModelId,
  onSelectModel,
  onRefresh,
}: ModelListProps) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const modelList = await listModels()
      setModels(modelList)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models')
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model?')) return

    try {
      await deleteModel(modelId)
      if (selectedModelId === modelId) {
        onSelectModel(models.find(m => m.model_id !== modelId)?.model_id || '')
      }
      fetchModels()
      onRefresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model')
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-gray-800">Trained Models</h2>
        <button
          onClick={fetchModels}
          className="text-sm text-blue-600 hover:text-blue-800"
        >
          Refresh
        </button>
      </div>

      {isLoading ? (
        <p className="text-gray-500 text-center py-4">Loading models...</p>
      ) : error ? (
        <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      ) : models.length === 0 ? (
        <p className="text-gray-500 text-center py-4">No models trained yet</p>
      ) : (
        <div className="space-y-2">
          {models.map((model) => (
            <div
              key={model.model_id}
              className={`p-3 border rounded-md cursor-pointer transition-colors ${
                selectedModelId === model.model_id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              }`}
              onClick={() => onSelectModel(model.model_id)}
            >
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-mono text-sm text-gray-800">
                    {model.model_id.slice(0, 8)}...
                  </p>
                  <p className="text-xs text-gray-500">
                    Layers: [{model.hidden_layers.join(', ')}]
                  </p>
                  {model.training_loss !== null && (
                    <p className="text-xs text-gray-500">
                      Loss: {model.training_loss.toFixed(6)}
                      {model.validation_loss !== null && (
                        <span> | Val: {model.validation_loss.toFixed(6)}</span>
                      )}
                    </p>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(model.model_id)
                  }}
                  className="text-red-500 hover:text-red-700 text-sm"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
