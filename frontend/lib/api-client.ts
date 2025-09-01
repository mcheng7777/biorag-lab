

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"

interface ApiErrorData {
  detail?: string
  [key: string]: unknown
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: ApiErrorData
  ) {
    super(message)
    this.name = "ApiError"
  }
}

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  
  const config: RequestInit = {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  }

  try {
    const response = await fetch(url, config)
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new ApiError(
        errorData.detail || `HTTP error! status: ${response.status}`,
        response.status,
        errorData
      )
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) {
      throw error
    }
    throw new ApiError(
      error instanceof Error ? error.message : "Network error",
      0
    )
  }
}

interface Paper {
  id: string
  title: string
  authors: string
  journal: string
  year: string
  abstract: string
}

interface Dataset {
  id: string
  title: string
  platform: string
  samples: string
  organism: string
  description: string
}

interface SearchResponse<T> {
  papers?: T[]
  datasets?: T[]
  message?: string
}

// API endpoints
export const api = {
  // Health check
  health: () => apiRequest<{ status: string; services: Record<string, string> }>("/health"),
  
  // Root endpoint
  root: () => apiRequest<{ status: string; message: string; version: string }>("/"),
  
  // Search papers (to be implemented)
  searchPapers: (query: string) => 
    apiRequest<SearchResponse<Paper>>(`/papers/search?q=${encodeURIComponent(query)}`),
  
  // Search datasets (to be implemented)
  searchDatasets: (query: string) => 
    apiRequest<SearchResponse<Dataset>>(`/datasets/search?q=${encodeURIComponent(query)}`),
  
  // Generate code (to be implemented)
  generateCode: (data: { prompt: string; language?: string }) =>
    apiRequest<{ code: string; explanation?: string }>("/generate_code", {
      method: "POST",
      body: JSON.stringify(data),
    }),
}

export { ApiError }
