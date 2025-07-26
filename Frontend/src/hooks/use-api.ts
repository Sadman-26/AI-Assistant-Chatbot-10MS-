import { useMutation, useQuery } from '@tanstack/react-query';
import { apiService, QuestionRequest, QuestionResponse, HealthResponse } from '@/services/api';

// Hook for health check
export const useHealthCheck = () => {
  return useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: () => apiService.healthCheck(),
    refetchInterval: 30000, // Refetch every 30 seconds
    retry: 3,
    retryDelay: 1000,
  });
};

// Hook for asking questions
export const useAskQuestion = () => {
  return useMutation<QuestionResponse, Error, QuestionRequest>({
    mutationFn: (request: QuestionRequest) => apiService.askQuestion(request),
    onError: (error) => {
      console.error('Failed to ask question:', error);
    },
  });
};

// Hook for getting API documentation
export const useApiDocumentation = () => {
  return useQuery({
    queryKey: ['api-docs'],
    queryFn: () => apiService.getDocumentation(),
    retry: 2,
    retryDelay: 1000,
  });
}; 