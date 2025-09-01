"use client"

import { useState } from "react"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { LoadingButton } from "@/components/ui/loading-button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { useSearchPapers } from "@/hooks/use-api"
import { useToast } from "@/components/ui/use-toast"

interface Paper {
  id: string
  title: string
  authors: string
  journal: string
  year: string
  abstract: string
}

export default function PapersPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState("all")
  const { toast } = useToast()
  
  const papersQuery = useSearchPapers(searchQuery, searchQuery.length > 0)

  const handleSearch = () => {
    if (!searchQuery.trim()) {
      toast({
        title: "Search Query Required",
        description: "Please enter a search query to continue.",
        variant: "destructive",
      })
      return
    }
    papersQuery.refetch()
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Research Papers"
        text="Search and explore papers from PMC and arXiv"
      >
        <div className="flex w-full max-w-[800px] items-center space-x-2">
          <Input 
            placeholder="Search papers..." 
            className="flex-1"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <LoadingButton 
            onClick={handleSearch}
            loading={papersQuery.isFetching}
            loadingText="Searching..."
          >
            Search
          </LoadingButton>
        </div>
      </PageHeader>

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList>
            <TabsTrigger value="all">All Papers</TabsTrigger>
            <TabsTrigger value="pmc">PMC</TabsTrigger>
            <TabsTrigger value="arxiv">arXiv</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="space-y-4">
            {papersQuery.isLoading && (
              <>
                <PaperCardSkeleton />
                <PaperCardSkeleton />
                <PaperCardSkeleton />
              </>
            )}
            
            {papersQuery.isError && (
              <Card>
                <CardContent className="p-6">
                  <p className="text-center text-muted-foreground">
                    Failed to load papers. Please try again.
                  </p>
                </CardContent>
              </Card>
            )}
            
            {papersQuery.isSuccess && papersQuery.data?.papers && papersQuery.data.papers.length === 0 && (
              <Card>
                <CardContent className="p-6">
                  <p className="text-center text-muted-foreground">
                    No papers found for your search query.
                  </p>
                </CardContent>
              </Card>
            )}
            
            {papersQuery.isSuccess && papersQuery.data?.papers && papersQuery.data.papers.length > 0 && (
              papersQuery.data.papers.map((paper: Paper, index: number) => (
                <PaperCard key={index} paper={paper} />
              ))
            )}
            
            {!papersQuery.isLoading && !papersQuery.isError && !papersQuery.isSuccess && (
              <PaperCard 
                paper={{
                  id: "1",
                  title: "Single-cell RNA sequencing: recent advances and challenges",
                  authors: "John Doe, Jane Smith",
                  journal: "Nature Methods",
                  year: "2023",
                  abstract: "This review discusses recent technological advances in single-cell RNA sequencing..."
                }} 
              />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

function PaperCard({ paper }: { paper: Paper }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{paper.title}</CardTitle>
        <CardDescription>
          {paper.authors} - Published in {paper.journal}, {paper.year}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">
          {paper.abstract}
        </p>
        <div className="flex items-center gap-2 mt-4">
          <LoadingButton variant="outline" size="sm">
            View Details
          </LoadingButton>
          <LoadingButton variant="outline" size="sm">
            Download PDF
          </LoadingButton>
        </div>
      </CardContent>
    </Card>
  )
}

function PaperCardSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-3/4 mb-2" />
        <Skeleton className="h-4 w-1/2" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-4 w-full mb-2" />
        <Skeleton className="h-4 w-full mb-2" />
        <Skeleton className="h-4 w-2/3 mb-4" />
        <div className="flex items-center gap-2">
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-8 w-24" />
        </div>
      </CardContent>
    </Card>
  )
}
