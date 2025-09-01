"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { Button } from "@/components/ui/button"
import { LoadingButton } from "@/components/ui/loading-button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useSearch } from "@/hooks/use-api"
import { useToast } from "@/components/ui/use-toast"

interface SearchFormData {
  query: string
}

export default function Home() {
  const [searchQuery, setSearchQuery] = useState("")
  const { register, handleSubmit, formState: { errors } } = useForm<SearchFormData>()
  const searchMutation = useSearch()
  const { toast } = useToast()

  const onSubmit = async (data: SearchFormData) => {
    if (!data.query.trim()) {
      toast({
        title: "Search Query Required",
        description: "Please enter a search query to continue.",
        variant: "destructive",
      })
      return
    }

    setSearchQuery(data.query)
    searchMutation.mutate(data.query)
  }

  return (
    <div className="flex flex-col items-center justify-start min-h-[calc(100vh-4rem)] space-y-8 py-8">
      <div className="text-center space-y-3 max-w-2xl">
        <h1 className="text-4xl font-bold tracking-tight">Welcome to BioRAG Lab</h1>
        <p className="text-xl text-muted-foreground">
          Discover papers, datasets, and generate code for bioinformatics research
        </p>
      </div>

      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle>Ask a Question</CardTitle>
          <CardDescription>
            Enter your research question or describe what you&apos;re looking for
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="space-y-2">
              <Textarea
                {...register("query", { 
                  required: "Search query is required",
                  minLength: { value: 3, message: "Query must be at least 3 characters" }
                })}
                placeholder="e.g., Find papers and datasets about single-cell RNA sequencing analysis methods..."
                className="min-h-[100px] resize-none"
              />
              {errors.query && (
                <p className="text-sm text-destructive">{errors.query.message}</p>
              )}
            </div>
            <div className="flex justify-end">
              <LoadingButton 
                type="submit" 
                loading={searchMutation.isPending}
                loadingText="Searching..."
              >
                Search
              </LoadingButton>
            </div>
          </form>
        </CardContent>
      </Card>

      {searchMutation.isSuccess && (
        <Card className="w-full max-w-2xl">
          <CardHeader>
            <CardTitle>Search Results</CardTitle>
            <CardDescription>
              Results for: &quot;{searchQuery}&quot;
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Search completed successfully! Results will be displayed here when the backend API is implemented.
            </p>
          </CardContent>
        </Card>
      )}

      <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Papers</CardTitle>
            <CardDescription>
              Search through PMC and arXiv papers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="outline" className="w-full" asChild>
              <a href="/papers">Browse Papers</a>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Datasets</CardTitle>
            <CardDescription>
              Find relevant GEO and SRA datasets
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="outline" className="w-full" asChild>
              <a href="/datasets">Browse Datasets</a>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Code Playground</CardTitle>
            <CardDescription>
              Generate and test bioinformatics code
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="outline" className="w-full" asChild>
              <a href="/playground">Open Playground</a>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}