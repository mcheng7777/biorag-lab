"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingButton } from "@/components/ui/loading-button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { useGenerateCode } from "@/hooks/use-api"
import { useToast } from "@/components/ui/use-toast"

interface DocsFormData {
  packageUrl: string
  prompt: string
}

interface PaperFormData {
  implementationType: string
  description: string
}

export default function PlaygroundPage() {
  const [activeTab, setActiveTab] = useState("docs")
  const [generatedCode, setGeneratedCode] = useState("")
  const [generatedExplanation, setGeneratedExplanation] = useState("")
  const { toast } = useToast()
  
  const docsForm = useForm<DocsFormData>()
  const paperForm = useForm<PaperFormData>()
  const generateCodeMutation = useGenerateCode()

  const handleDocsSubmit = async (data: DocsFormData) => {
    if (!data.packageUrl.trim() || !data.prompt.trim()) {
      toast({
        title: "Missing Information",
        description: "Please fill in both package URL and prompt fields.",
        variant: "destructive",
      })
      return
    }

    generateCodeMutation.mutate(
      { 
        prompt: `Package: ${data.packageUrl}\n\nRequest: ${data.prompt}`,
        language: "python" // Default to Python for docs
      },
      {
        onSuccess: (result) => {
          setGeneratedCode(result.code || "// Code will be generated here...")
          setGeneratedExplanation(result.explanation || "Explanation will appear here...")
        }
      }
    )
  }

  const handlePaperSubmit = async (data: PaperFormData) => {
    if (!data.description.trim()) {
      toast({
        title: "Missing Description",
        description: "Please provide a description of what you want to implement.",
        variant: "destructive",
      })
      return
    }

    generateCodeMutation.mutate(
      { 
        prompt: `Implementation Type: ${data.implementationType}\n\nDescription: ${data.description}`,
        language: "python" // Default to Python for paper implementations
      },
      {
        onSuccess: (result) => {
          setGeneratedCode(result.code || "// Implementation will be generated here...")
          setGeneratedExplanation(result.explanation || "Method explanation will appear here...")
        }
      }
    )
  }

  const handleClear = () => {
    setGeneratedCode("")
    setGeneratedExplanation("")
    docsForm.reset()
    paperForm.reset()
  }

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Code Playground"
        text="Generate and test bioinformatics code using Gemini"
      />

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="docs">Package Documentation</TabsTrigger>
            <TabsTrigger value="paper">Paper Implementation</TabsTrigger>
          </TabsList>

          {/* Documentation-Based Code Generation */}
          <TabsContent value="docs">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="lg:h-[calc(100vh-16rem)]">
                <CardHeader>
                  <CardTitle>Package Documentation</CardTitle>
                  <CardDescription>
                    Generate code examples from package documentation
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={docsForm.handleSubmit(handleDocsSubmit)} className="space-y-4">
                    <div className="space-y-2">
                      <Input 
                        {...docsForm.register("packageUrl", { required: "Package URL is required" })}
                        placeholder="Package documentation URL (e.g., ComplexHeatmap, seaborn)"
                        className="w-full"
                      />
                      {docsForm.formState.errors.packageUrl && (
                        <p className="text-sm text-destructive">{docsForm.formState.errors.packageUrl.message}</p>
                      )}
                      <Textarea
                        {...docsForm.register("prompt", { required: "Prompt is required" })}
                        placeholder="What would you like to do with this package? (e.g., Create a heatmap with row/column clustering)"
                        className="min-h-[150px]"
                      />
                      {docsForm.formState.errors.prompt && (
                        <p className="text-sm text-destructive">{docsForm.formState.errors.prompt.message}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <LoadingButton 
                        type="submit"
                        loading={generateCodeMutation.isPending}
                        loadingText="Generating..."
                      >
                        Generate Code
                      </LoadingButton>
                      <LoadingButton 
                        type="button" 
                        variant="outline" 
                        onClick={handleClear}
                        disabled={generateCodeMutation.isPending}
                      >
                        Clear
                      </LoadingButton>
                    </div>
                  </form>
                </CardContent>
              </Card>

              <Card className="lg:h-[calc(100vh-16rem)]">
                <CardHeader>
                  <CardTitle>Generated Code</CardTitle>
                  <CardDescription>
                    Choose your preferred programming language
                  </CardDescription>
                  <Tabs defaultValue="r" className="w-full">
                    <TabsList>
                      <TabsTrigger value="r">R</TabsTrigger>
                      <TabsTrigger value="python">Python</TabsTrigger>
                    </TabsList>
                    <TabsContent value="r" className="mt-4">
                      <Card>
                        <CardContent className="p-4">
                          {generateCodeMutation.isPending ? (
                            <div className="space-y-2">
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-3/4" />
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-2/3" />
                            </div>
                          ) : (
                            <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                              <code>
                                {generatedCode || `# Example R code with ComplexHeatmap
library(ComplexHeatmap)
library(circlize)

# Create example matrix
set.seed(123)
mat = matrix(rnorm(100), 10)
rownames(mat) = paste0("Gene", 1:10)
colnames(mat) = paste0("Sample", 1:10)

# Create heatmap
Heatmap(mat,
  name = "Expression",
  row_title = "Genes",
  column_title = "Samples",
  clustering_distance_rows = "euclidean",
  clustering_method_rows = "complete"
)`}
                              </code>
                            </pre>
                          )}
                        </CardContent>
                      </Card>
                    </TabsContent>
                  </Tabs>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <LoadingButton variant="outline" disabled={!generatedCode}>
                        Copy Code
                      </LoadingButton>
                      <LoadingButton variant="outline" disabled={!generatedCode}>
                        Download
                      </LoadingButton>
                      <LoadingButton variant="outline" disabled={!generatedCode}>
                        Run Code
                      </LoadingButton>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Paper Implementation */}
          <TabsContent value="paper">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="lg:h-[calc(100vh-16rem)]">
                <CardHeader>
                  <CardTitle>Paper Implementation</CardTitle>
                  <CardDescription>
                    Generate code to implement methods from papers
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={paperForm.handleSubmit(handlePaperSubmit)} className="space-y-4">
                    <Card className="bg-muted">
                      <CardHeader>
                        <CardTitle className="text-sm">Selected Paper</CardTitle>
                        <CardDescription className="text-xs">
                          Choose a paper from the Papers section
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <LoadingButton variant="outline" size="sm" className="w-full">
                          Select Paper
                        </LoadingButton>
                      </CardContent>
                    </Card>

                    <Card className="bg-muted">
                      <CardHeader>
                        <CardTitle className="text-sm">Selected Dataset</CardTitle>
                        <CardDescription className="text-xs">
                          Choose a dataset from the Datasets section
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <LoadingButton variant="outline" size="sm" className="w-full">
                          Select Dataset
                        </LoadingButton>
                      </CardContent>
                    </Card>

                    <div className="space-y-2">
                      <label className="text-sm font-medium">Implementation Type</label>
                      <select 
                        {...paperForm.register("implementationType")}
                        className="w-full rounded-md border border-input bg-background px-3 py-2"
                      >
                        <option value="figure">Figure/Plot Reproduction</option>
                        <option value="analysis">Statistical Analysis</option>
                        <option value="method">Method Implementation</option>
                      </select>
                    </div>

                    <Textarea
                      {...paperForm.register("description", { required: "Description is required" })}
                      placeholder="Describe what aspect of the paper you want to implement..."
                      className="min-h-[100px]"
                    />
                    {paperForm.formState.errors.description && (
                      <p className="text-sm text-destructive">{paperForm.formState.errors.description.message}</p>
                    )}

                    <div className="flex items-center gap-2">
                      <LoadingButton 
                        type="submit"
                        loading={generateCodeMutation.isPending}
                        loadingText="Generating..."
                      >
                        Generate Implementation
                      </LoadingButton>
                      <LoadingButton 
                        type="button" 
                        variant="outline" 
                        onClick={handleClear}
                        disabled={generateCodeMutation.isPending}
                      >
                        Clear
                      </LoadingButton>
                    </div>
                  </form>
                </CardContent>
              </Card>

              <Card className="lg:h-[calc(100vh-16rem)]">
                <CardHeader>
                  <CardTitle>Generated Implementation</CardTitle>
                  <CardDescription>
                    Code implementation with explanations
                  </CardDescription>
                  <Tabs defaultValue="code" className="w-full">
                    <TabsList>
                      <TabsTrigger value="code">Code</TabsTrigger>
                      <TabsTrigger value="explanation">Explanation</TabsTrigger>
                    </TabsList>
                    <TabsContent value="code" className="mt-4">
                      <Card>
                        <CardContent className="p-4">
                          {generateCodeMutation.isPending ? (
                            <div className="space-y-2">
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-3/4" />
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-2/3" />
                            </div>
                          ) : (
                            <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                              <code>
                                {generatedCode || `# Implementation will appear here...`}
                              </code>
                            </pre>
                          )}
                        </CardContent>
                      </Card>
                    </TabsContent>
                    <TabsContent value="explanation" className="mt-4">
                      <Card>
                        <CardContent className="p-4">
                          {generateCodeMutation.isPending ? (
                            <div className="space-y-2">
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-3/4" />
                            </div>
                          ) : (
                            <div className="prose prose-sm">
                              <p>{generatedExplanation || "Method explanation will appear here..."}</p>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    </TabsContent>
                  </Tabs>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <LoadingButton variant="outline" disabled={!generatedCode}>
                        Copy Code
                      </LoadingButton>
                      <LoadingButton variant="outline" disabled={!generatedCode}>
                        Download
                      </LoadingButton>
                      <LoadingButton variant="outline" disabled={!generatedCode}>
                        Run Code
                      </LoadingButton>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}