import * as React from "react"
import { Button } from "@/components/ui/button"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { cn } from "@/lib/utils"

interface LoadingButtonProps extends React.ComponentProps<typeof Button> {
  loading?: boolean
  loadingText?: string
  children: React.ReactNode
}

export const LoadingButton = React.forwardRef<
  HTMLButtonElement,
  LoadingButtonProps
>(({ loading = false, loadingText, children, disabled, className, ...props }, ref) => {
  return (
    <Button
      ref={ref}
      disabled={disabled || loading}
      className={cn(className)}
      {...props}
    >
      {loading && (
        <LoadingSpinner size="sm" className="mr-2" />
      )}
      {loading && loadingText ? loadingText : children}
    </Button>
  )
})

LoadingButton.displayName = "LoadingButton"
