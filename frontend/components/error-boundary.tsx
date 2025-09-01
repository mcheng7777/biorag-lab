"use client"

import { ErrorBoundary as ReactErrorBoundary } from "react-error-boundary"
import { ErrorFallback } from "@/components/error-fallback"

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ComponentType<{ error: Error; resetErrorBoundary: () => void }>
}

export function ErrorBoundary({ 
  children, 
  fallback: FallbackComponent = ErrorFallback 
}: ErrorBoundaryProps) {
  return (
    <ReactErrorBoundary
      FallbackComponent={FallbackComponent}
      onError={(error, errorInfo) => {
        // Log error to console in development
        if (process.env.NODE_ENV === "development") {
          console.error("Error caught by boundary:", error, errorInfo)
        }
        // In production, you might want to send this to an error reporting service
        // like Sentry, LogRocket, etc.
      }}
    >
      {children}
    </ReactErrorBoundary>
  )
}
