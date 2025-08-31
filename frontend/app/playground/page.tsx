import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"

export default function PlaygroundPage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Code Playground"
        text="Generate and test bioinformatics code using Gemini"
      />

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <Tabs defaultValue="docs" className="w-full">
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
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Input 
                        placeholder="Package documentation URL (e.g., ComplexHeatmap, seaborn)"
                        className="w-full"
                      />
                      <Textarea
                        placeholder="What would you like to do with this package? (e.g., Create a heatmap with row/column clustering)"
                        className="min-h-[150px]"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <Button>Generate Code</Button>
                      <Button variant="outline">Clear</Button>
                    </div>
                  </div>
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
                          <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                            <code>
                              {`# Example R code with ComplexHeatmap
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
                        </CardContent>
                      </Card>
                    </TabsContent>
                  </Tabs>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Button variant="outline">Copy Code</Button>
                      <Button variant="outline">Download</Button>
                      <Button variant="outline">Run Code</Button>
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
                  <div className="space-y-4">
                    <Card className="bg-muted">
                      <CardHeader>
                        <CardTitle className="text-sm">Selected Paper</CardTitle>
                        <CardDescription className="text-xs">
                          Choose a paper from the Papers section
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Button variant="outline" size="sm" className="w-full">
                          Select Paper
                        </Button>
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
                        <Button variant="outline" size="sm" className="w-full">
                          Select Dataset
                        </Button>
                      </CardContent>
                    </Card>

                    <div className="space-y-2">
                      <label className="text-sm font-medium">Implementation Type</label>
                      <select className="w-full rounded-md border border-input bg-background px-3 py-2">
                        <option>Figure/Plot Reproduction</option>
                        <option>Statistical Analysis</option>
                        <option>Method Implementation</option>
                      </select>
                    </div>

                    <Textarea
                      placeholder="Describe what aspect of the paper you want to implement..."
                      className="min-h-[100px]"
                    />

                    <div className="flex items-center gap-2">
                      <Button>Generate Implementation</Button>
                      <Button variant="outline">Clear</Button>
                    </div>
                  </div>
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
                          <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                            <code>
                              {`# Implementation will appear here...`}
                            </code>
                          </pre>
                        </CardContent>
                      </Card>
                    </TabsContent>
                    <TabsContent value="explanation" className="mt-4">
                      <Card>
                        <CardContent className="p-4">
                          <div className="prose prose-sm">
                            <p>Method explanation will appear here...</p>
                          </div>
                        </CardContent>
                      </Card>
                    </TabsContent>
                  </Tabs>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Button variant="outline">Copy Code</Button>
                      <Button variant="outline">Download</Button>
                      <Button variant="outline">Run Code</Button>
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