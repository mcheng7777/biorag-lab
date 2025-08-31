import Link from "next/link"
import { Button } from "@/components/ui/button"

export function MainNav() {
  return (
    <nav className="flex items-center space-x-4 lg:space-x-6">
      <Link 
        href="/"
        className="text-sm font-medium transition-colors hover:text-primary"
      >
        Home
      </Link>
      <Link
        href="/papers"
        className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
      >
        Papers
      </Link>
      <Link
        href="/datasets"
        className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
      >
        Datasets
      </Link>
      <Link
        href="/playground"
        className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
      >
        Code Playground
      </Link>
    </nav>
  )
}
