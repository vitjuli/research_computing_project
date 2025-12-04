'use client'

import { useState } from 'react'
import { predictSingle, predictBatch } from '@/lib/api'

interface PredictionFormProps {
  modelId: string | null
}

export default function PredictionForm({ modelId }: PredictionFormProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'single' | 'batch'>('single')

  // Single prediction inputs
  const [x1, setX1] = useState(0.5)
  const [x2, setX2] = useState(0.5)
  const [x3, setX3] = useState(0.5)
  const [x4, setX4] = useState(0.5)
  const [x5, setX5] = useState(0.5)

  // Batch prediction input
  const [batchInput, setBatchInput] = useState('')

  // Results
  const [singleResult, setSingleResult] = useState<number | null>(null)
  const [batchResults, setBatchResults] = useState<number[][] | null>(null)

  const handleSinglePredict = async () => {
    if (!modelId) return

    setIsLoading(true)
    setError(null)
    setSingleResult(null)

    try {
      const response = await predictSingle(modelId, { x1, x2, x3, x4, x5 })
      setSingleResult(response.prediction)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setIsLoading(false)
    }
  }

  const handleBatchPredict = async () => {
    if (!modelId) return

    setIsLoading(true)
    setError(null)
    setBatchResults(null)

    try {
      // Parse batch input
      const lines = batchInput.trim().split('\n').filter(line => line.trim())
      if (lines.length === 0) {
        throw new Error('No data provided')
      }

      const X: number[][] = []
      for (const line of lines) {
        const values = line.split(',').map(v => parseFloat(v.trim()))
        if (values.length !== 5) {
          throw new Error('Each line must have exactly 5 values')
        }
        X.push(values)
      }

      const response = await predictBatch(modelId, { X })
      setBatchResults(response.predictions)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setIsLoading(false)
    }
  }

  const generateRandomInputs = () => {
    setX1(Math.random())
    setX2(Math.random())
    setX3(Math.random())
    setX4(Math.random())
    setX5(Math.random())
  }

  if (!modelId) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800">Make Predictions</h2>
        <p className="text-gray-500 text-center py-8">
          Train or select a model to make predictions
        </p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Make Predictions</h2>

      <div className="mb-4">
        <p className="text-sm text-gray-600">
          Using model: <span className="font-mono text-blue-600">{modelId.slice(0, 8)}...</span>
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => setMode('single')}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            mode === 'single'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Single Point
        </button>
        <button
          onClick={() => setMode('batch')}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            mode === 'batch'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Batch Prediction
        </button>
      </div>

      {mode === 'single' ? (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-700">Input Values</span>
            <button
              type="button"
              onClick={generateRandomInputs}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Randomize
            </button>
          </div>

          <div className="grid grid-cols-5 gap-2">
            {[
              { label: 'x1', value: x1, setter: setX1 },
              { label: 'x2', value: x2, setter: setX2 },
              { label: 'x3', value: x3, setter: setX3 },
              { label: 'x4', value: x4, setter: setX4 },
              { label: 'x5', value: x5, setter: setX5 },
            ].map(({ label, value, setter }) => (
              <div key={label}>
                <label className="block text-xs font-medium text-gray-500 mb-1 text-center">
                  {label}
                </label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => setter(parseFloat(e.target.value) || 0)}
                  step={0.1}
                  className="w-full px-2 py-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm text-center"
                />
              </div>
            ))}
          </div>

          <button
            onClick={handleSinglePredict}
            disabled={isLoading}
            className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Predicting...' : 'Predict'}
          </button>

          {singleResult !== null && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-gray-600">Prediction Result:</p>
              <p className="text-2xl font-bold text-blue-700">{singleResult.toFixed(6)}</p>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Batch Input (CSV format: x1,x2,x3,x4,x5)
            </label>
            <textarea
              value={batchInput}
              onChange={(e) => setBatchInput(e.target.value)}
              rows={6}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
              placeholder={"0.1,0.2,0.3,0.4,0.5\n0.2,0.3,0.4,0.5,0.6\n..."}
            />
          </div>

          <button
            onClick={handleBatchPredict}
            disabled={isLoading || !batchInput.trim()}
            className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Predicting...' : 'Predict Batch'}
          </button>

          {batchResults && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-gray-600 mb-2">Predictions:</p>
              <div className="max-h-48 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-1">#</th>
                      <th className="text-right py-1">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((pred, i) => (
                      <tr key={i} className="border-b border-gray-200">
                        <td className="py-1">{i + 1}</td>
                        <td className="text-right font-mono">{pred[0].toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}
    </div>
  )
}
