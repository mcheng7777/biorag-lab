"use client"

import { useState } from "react"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { LoadingButton } from "@/components/ui/loading-button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { useSearchDatasets } from "@/hooks/use-api"
import { useToast } from "@/components/ui/use-toast"

interface Dataset {
  id: string
  title: string
  platform: string
  samples: string
  organism: string
  description: string
}

export default function DatasetsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState("all")
  const { toast } = useToast()
  
  const datasetsQuery = useSearchDatasets(searchQuery, searchQuery.length > 0)

  const handleSearch = () => {
    if (!searchQuery.trim()) {
      toast({
        title: "Search Query Required",
        description: "Please enter a search query to continue.",
        variant: "destructive",
      })
      return
    }
    datasetsQuery.refetch()
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Datasets"
        text="Find relevant datasets from GEO and SRA databases"
      >
        <div className="flex w-full max-w-[800px] items-center space-x-2">
          <Input 
            placeholder="Search datasets..." 
            className="flex-1"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <LoadingButton 
            onClick={handleSearch}
            loading={datasetsQuery.isFetching}
            loadingText="Searching..."
          >
            Search
          </LoadingButton>
        </div>
      </PageHeader>

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList>
            <TabsTrigger value="all">All Datasets</TabsTrigger>
            <TabsTrigger value="geo">GEO</TabsTrigger>
            <TabsTrigger value="sra">SRA</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="space-y-4">
            {datasetsQuery.isLoading && (
              <>
                <DatasetCardSkeleton />
                <DatasetCardSkeleton />
                <DatasetCardSkeleton />
              </>
            )}
            
            {datasetsQuery.isError && (
              <Card>
                <CardContent className="p-6">
                  <p className="text-center text-muted-foreground">
                    Failed to load datasets. Please try again.
                  </p>
                </CardContent>
              </Card>
            )}
            
            {datasetsQuery.isSuccess && datasetsQuery.data?.datasets && datasetsQuery.data.datasets.length === 0 && (
              <Card>
                <CardContent className="p-6">
                  <p className="text-center text-muted-foreground">
                    No datasets found for your search query.
                  </p>
                </CardContent>
              </Card>
            )}
            
            {datasetsQuery.isSuccess && datasetsQuery.data?.datasets && datasetsQuery.data.datasets.length > 0 && (
              datasetsQuery.data.datasets.map((dataset: Dataset, index: number) => (
                <DatasetCard key={index} dataset={dataset} />
              ))
            )}
            
            {!datasetsQuery.isLoading && !datasetsQuery.isError && !datasetsQuery.isSuccess && (
              <DatasetCard 
                dataset={{
                  id: "GSE123456",
                  title: "Single-cell transcriptomics of human T cells",
                  platform: "Illumina NovaSeq 6000",
                  samples: "10",
                  organism: "Homo sapiens",
                  description: "This dataset contains single-cell RNA sequencing data from human T cells..."
                }} 
              />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

function DatasetCard({ dataset }: { dataset: Dataset }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{dataset.id}: {dataset.title}</CardTitle>
        <CardDescription>
          Platform: {dataset.platform} - Samples: {dataset.samples} - Organism: {dataset.organism}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">
          {dataset.description}
        </p>
        <div className="flex items-center gap-2 mt-4">
          <LoadingButton variant="outline" size="sm">
            View Details
          </LoadingButton>
          <LoadingButton variant="outline" size="sm">
            Download Data
          </LoadingButton>
          <LoadingButton variant="outline" size="sm">
            View Analysis Code
          </LoadingButton>
        </div>
      </CardContent>
    </Card>
  )
}

function DatasetCardSkeleton() {
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
          <Skeleton className="h-8 w-32" />
        </div>
      </CardContent>
    </Card>
  )
}
