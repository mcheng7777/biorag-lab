import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"

export default function PlaygroundPage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Code Playground"
        text="Generate and test bioinformatics code using Gemini"
      />

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="lg:h-[calc(100vh-16rem)]">
            <CardHeader>
              <CardTitle>Input</CardTitle>
              <CardDescription>
                Describe what you want to do, and we&apos;ll generate the code
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Textarea
                  placeholder="e.g., Generate code to perform differential expression analysis on RNA-seq data..."
                  className="min-h-[200px] lg:min-h-[400px]"
                />
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
                          {`# Example R code
library(DESeq2)
library(tidyverse)

# Read count data
counts <- read.csv("counts.csv", row.names=1)
metadata <- read.csv("metadata.csv", row.names=1)

# Create DESeq object
dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = metadata,
  design = ~ condition
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
      </div>
    </div>
  )
}
