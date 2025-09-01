"use client"

import { useQuery, useMutation } from "@tanstack/react-query"
import { api, ApiError } from "@/lib/api-client"
import { useToast } from "@/components/ui/use-toast"

// Health check hook
export function useHealthCheck() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Search papers hook
export function useSearchPapers(query: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["papers", "search", query],
    queryFn: () => api.searchPapers(query),
    enabled: enabled && query.length > 0,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

// Search datasets hook
export function useSearchDatasets(query: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["datasets", "search", query],
    queryFn: () => api.searchDatasets(query),
    enabled: enabled && query.length > 0,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

// Generate code hook
export function useGenerateCode() {
  const { toast } = useToast()
  
  return useMutation({
    mutationFn: api.generateCode,
    onSuccess: () => {
      toast({
        title: "Code Generated",
        description: "Your code has been generated successfully!",
      })
    },
    onError: (error: ApiError) => {
      toast({
        title: "Generation Failed",
        description: error.message || "Failed to generate code. Please try again.",
        variant: "destructive",
      })
    },
  })
}

// Generic search hook for the home page
export function useSearch() {
  const { toast } = useToast()
  
  return useMutation({
    mutationFn: async (_: string) => { // eslint-disable-line @typescript-eslint/no-unused-vars
      // For now, just return a mock response
      // This will be replaced with actual API calls later
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            papers: [],
            datasets: [],
            message: "Search completed",
          })
        }, 1000)
      })
    },
    onSuccess: () => {
      toast({
        title: "Search Completed",
        description: "Your search has been completed successfully!",
      })
    },
    onError: (error: Error) => {
      toast({
        title: "Search Failed",
        description: error.message || "Failed to perform search. Please try again.",
        variant: "destructive",
      })
    },
  })
}
