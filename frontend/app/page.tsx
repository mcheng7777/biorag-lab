import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

export default function Home() {
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
          <form className="space-y-4">
            <Textarea
              placeholder="e.g., Find papers and datasets about single-cell RNA sequencing analysis methods..."
              className="min-h-[100px] resize-none"
            />
            <div className="flex justify-end">
              <Button type="submit">Search</Button>
            </div>
          </form>
        </CardContent>
      </Card>

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