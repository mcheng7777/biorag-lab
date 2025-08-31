import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function PapersPage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <PageHeader
        heading="Research Papers"
        text="Search and explore papers from PMC and arXiv"
      >
        <div className="flex w-full max-w-[800px] items-center space-x-2">
          <Input placeholder="Search papers..." className="flex-1" />
          <Button>Search</Button>
        </div>
      </PageHeader>

      <div className="flex-1 space-y-6 px-4 sm:px-6 lg:px-8">
        <Tabs defaultValue="all" className="w-full">
          <TabsList>
            <TabsTrigger value="all">All Papers</TabsTrigger>
            <TabsTrigger value="pmc">PMC</TabsTrigger>
            <TabsTrigger value="arxiv">arXiv</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="space-y-4">
            {/* Example paper card - we'll make this dynamic later */}
            <Card>
              <CardHeader>
                <CardTitle>Single-cell RNA sequencing: recent advances and challenges</CardTitle>
                <CardDescription>Authors: John Doe, Jane Smith - Published in Nature Methods, 2023</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  This review discusses recent technological advances in single-cell RNA sequencing...
                </p>
                <div className="flex items-center gap-2 mt-4">
                  <Button variant="outline" size="sm">View Details</Button>
                  <Button variant="outline" size="sm">Download PDF</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
