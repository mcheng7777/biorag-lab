import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function DatasetsPage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Datasets"
        text="Find relevant datasets from GEO and SRA databases"
      >
        <div className="flex w-full max-w-[800px] items-center space-x-2">
          <Input placeholder="Search datasets..." className="flex-1" />
          <Button>Search</Button>
        </div>
      </PageHeader>

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <Tabs defaultValue="all" className="w-full">
          <TabsList>
            <TabsTrigger value="all">All Datasets</TabsTrigger>
            <TabsTrigger value="geo">GEO</TabsTrigger>
            <TabsTrigger value="sra">SRA</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="space-y-4">
            {/* Example dataset card - we'll make this dynamic later */}
            <Card>
              <CardHeader>
                <CardTitle>GSE123456: Single-cell transcriptomics of human T cells</CardTitle>
                <CardDescription>Platform: Illumina NovaSeq 6000 - Samples: 10 - Organism: Homo sapiens</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  This dataset contains single-cell RNA sequencing data from human T cells...
                </p>
                <div className="flex items-center gap-2 mt-4">
                  <Button variant="outline" size="sm">View Details</Button>
                  <Button variant="outline" size="sm">Download Data</Button>
                  <Button variant="outline" size="sm">View Analysis Code</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
