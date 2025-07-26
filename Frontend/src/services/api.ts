// API service for communicating with the backend FastAPI server

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface QuestionRequest {
  question: string;
  include_context?: boolean;
}

export interface QuestionResponse {
  answer: string;
  context?: string;
  success: boolean;
  message?: string;
}

export interface HealthResponse {
  status: string;
  message: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // Health check endpoint
  async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  }

  // Ask a question endpoint
  async askQuestion(request: QuestionRequest): Promise<QuestionResponse> {
    const response = await fetch(`${this.baseUrl}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Get API documentation
  async getDocumentation() {
    const response = await fetch(`${this.baseUrl}/docs`);
    if (!response.ok) {
      throw new Error(`Failed to get documentation: ${response.statusText}`);
    }
    return response.json();
  }
}

// Create and export a default instance
export const apiService = new ApiService();

// Export the class for testing or custom instances
export default ApiService; 