'use client'

import { useState } from 'react'
import TrainingForm from '@/components/TrainingForm'
import PredictionForm from '@/components/PredictionForm'
import ModelList from '@/components/ModelList'

export default function Home() {
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  const handleModelTrained = (modelId: string) => {
    setSelectedModelId(modelId)
    setRefreshKey(prev => prev + 1)
  }

  return (
    <main className="min-h-screen p-8 bg-gray-100">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          NN Interpolator
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Train and query neural network models for 5D numerical data interpolation
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Training */}
          <div className="space-y-8">
            <TrainingForm onModelTrained={handleModelTrained} />
            <ModelList
              key={refreshKey}
              selectedModelId={selectedModelId}
              onSelectModel={setSelectedModelId}
              onRefresh={() => setRefreshKey(prev => prev + 1)}
            />
          </div>

          {/* Right Column: Prediction */}
          <div>
            <PredictionForm modelId={selectedModelId} />
          </div>
        </div>
      </div>
    </main>
  )
}
