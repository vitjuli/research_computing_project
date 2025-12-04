const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface TrainingConfig {
  hidden_layers: number[]
  epochs: number
  batch_size: number
  learning_rate: number
  test_size: number
  normalize: boolean
}

export interface TrainingData {
  X: number[][]
  y: number[]
  config: TrainingConfig
}

export interface TrainingResponse {
  model_id: string
  message: string
  final_loss: number
  final_val_loss: number | null
  history: {
    loss: number[]
    val_loss: number[]
  }
}

export interface ModelInfo {
  model_id: string
  hidden_layers: number[]
  output_dim: number
  training_loss: number | null
  validation_loss: number | null
}

export interface PredictionInput {
  X: number[][]
}

export interface PredictionResponse {
  predictions: number[][]
}

export interface SinglePredictionInput {
  x1: number
  x2: number
  x3: number
  x4: number
  x5: number
}

export interface SinglePredictionResponse {
  prediction: number
}

// API Functions
export async function trainModel(data: TrainingData): Promise<TrainingResponse> {
  const response = await fetch(`${API_BASE_URL}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Training failed')
  }

  return response.json()
}

export async function listModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE_URL}/models`)

  if (!response.ok) {
    throw new Error('Failed to fetch models')
  }

  return response.json()
}

export async function getModel(modelId: string): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/models/${modelId}`)

  if (!response.ok) {
    throw new Error('Model not found')
  }

  return response.json()
}

export async function predictBatch(
  modelId: string,
  data: PredictionInput
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/models/${modelId}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Prediction failed')
  }

  return response.json()
}

export async function predictSingle(
  modelId: string,
  data: SinglePredictionInput
): Promise<SinglePredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/models/${modelId}/predict/single`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Prediction failed')
  }

  return response.json()
}

export async function deleteModel(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/models/${modelId}`, {
    method: 'DELETE',
  })

  if (!response.ok) {
    throw new Error('Failed to delete model')
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
