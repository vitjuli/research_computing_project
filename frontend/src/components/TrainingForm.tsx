'use client'

import { useState } from 'react'
import { trainModel, TrainingData } from '@/lib/api'

interface TrainingFormProps {
  onModelTrained: (modelId: string) => void
}

export default function TrainingForm({ onModelTrained }: TrainingFormProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{
    modelId: string
    finalLoss: number
    valLoss: number | null
  } | null>(null)

  // Training configuration
  const [hiddenLayers, setHiddenLayers] = useState('64,64,32')
  const [epochs, setEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.001)
  const [testSize, setTestSize] = useState(0.2)
  const [normalize, setNormalize] = useState(true)

  // Data input
  const [dataInput, setDataInput] = useState('')

  const generateSampleData = () => {
    // Generate sample data: X is 5D, y is sum of X
    const samples = 200
    const X: number[][] = []
    const y: number[] = []

    for (let i = 0; i < samples; i++) {
      const row = Array.from({ length: 5 }, () => Math.random())
      X.push(row)
      y.push(row.reduce((a, b) => a + b, 0))
    }

    const dataStr = X.map((row, i) => [...row, y[i]].join(',')).join('\n')
    setDataInput(dataStr)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      // Parse data input
      const lines = dataInput.trim().split('\n').filter(line => line.trim())
      if (lines.length === 0) {
        throw new Error('No data provided')
      }

      const X: number[][] = []
      const y: number[] = []

      for (const line of lines) {
        const values = line.split(',').map(v => parseFloat(v.trim()))
        if (values.length < 6) {
          throw new Error('Each line must have at least 6 values (5 inputs + 1 target)')
        }
        X.push(values.slice(0, 5))
        y.push(values[5])
      }

      // Parse hidden layers
      const layers = hiddenLayers.split(',').map(v => parseInt(v.trim()))
      if (layers.some(isNaN)) {
        throw new Error('Invalid hidden layers format')
      }

      const trainingData: TrainingData = {
        X,
        y,
        config: {
          hidden_layers: layers,
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          test_size: testSize,
          normalize,
        },
      }

      const response = await trainModel(trainingData)

      setResult({
        modelId: response.model_id,
        finalLoss: response.final_loss,
        valLoss: response.final_val_loss,
      })

      onModelTrained(response.model_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Train Model</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Data Input */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <label className="block text-sm font-medium text-gray-700">
              Training Data (CSV format: x1,x2,x3,x4,x5,y)
            </label>
            <button
              type="button"
              onClick={generateSampleData}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Generate Sample Data
            </button>
          </div>
          <textarea
            value={dataInput}
            onChange={(e) => setDataInput(e.target.value)}
            rows={6}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
            placeholder={"0.1,0.2,0.3,0.4,0.5,1.5\n0.2,0.3,0.4,0.5,0.6,2.0\n..."}
          />
        </div>

        {/* Configuration */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Hidden Layers
            </label>
            <input
              type="text"
              value={hiddenLayers}
              onChange={(e) => setHiddenLayers(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="64,64,32"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Epochs
            </label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min={1}
              max={10000}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Batch Size
            </label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              min={1}
              max={1024}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Learning Rate
            </label>
            <input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              step={0.001}
              min={0.0001}
              max={1}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Validation Split
            </label>
            <input
              type="number"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              step={0.05}
              min={0.05}
              max={0.5}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="normalize"
              checked={normalize}
              onChange={(e) => setNormalize(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="normalize" className="ml-2 block text-sm text-gray-700">
              Normalize Data
            </label>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading || !dataInput.trim()}
          className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Training...' : 'Train Model'}
        </button>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="mt-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded">
          <p><strong>Model trained successfully!</strong></p>
          <p>Model ID: {result.modelId}</p>
          <p>Final Loss: {result.finalLoss.toFixed(6)}</p>
          {result.valLoss !== null && (
            <p>Validation Loss: {result.valLoss.toFixed(6)}</p>
          )}
        </div>
      )}
    </div>
  )
}
