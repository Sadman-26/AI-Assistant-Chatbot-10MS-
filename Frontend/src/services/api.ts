// API service for communicating with the backend FastAPI server

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Debug logging
console.log('API_BASE_URL:', API_BASE_URL);
console.log('Environment variables:', import.meta.env);

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
    console.log('ApiService initialized with baseUrl:', this.baseUrl);
  }

  // Health check endpoint
  async healthCheck(): Promise<HealthResponse> {
    try {
      console.log('Attempting health check to:', `${this.baseUrl}/health`);
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      console.log('Health check response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Health check successful:', data);
      return data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  // Ask a question endpoint
  async askQuestion(request: QuestionRequest): Promise<QuestionResponse> {
    try {
      console.log('Attempting to ask question to:', `${this.baseUrl}/ask`);
      const response = await fetch(`${this.baseUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      console.log('Ask question response status:', response.status);

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Ask question successful:', data);
      return data;
    } catch (error) {
      console.error('Ask question error:', error);
      throw error;
    }
  }

  // Get API documentation
  async getDocumentation() {
    try {
      console.log('Attempting to get documentation from:', `${this.baseUrl}/docs`);
      const response = await fetch(`${this.baseUrl}/docs`);
      
      console.log('Documentation response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Failed to get documentation: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Documentation successful:', data);
      return data;
    } catch (error) {
      console.error('Documentation error:', error);
      throw error;
    }
  }
}

// Create and export a default instance
export const apiService = new ApiService();

// Export the class for testing or custom instances
export default ApiService; 