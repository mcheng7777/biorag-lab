import { MainNav } from "@/components/layout/main-nav"
import { Button } from "@/components/ui/button"

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mr-4 flex">
          <a href="/" className="mr-6 flex items-center space-x-2">
            <span className="font-bold">BioRAG Lab</span>
          </a>
        </div>
        <MainNav />
        <div className="ml-auto flex items-center space-x-4">
          {/* We'll add auth buttons here later */}
          <Button variant="outline">Sign In</Button>
        </div>
      </div>
    </header>
  )
}
