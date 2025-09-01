"use client"

import { Inter } from "next/font/google"
import "./globals.css"
import { Header } from "@/components/layout/header"
import { QueryProvider } from "@/components/providers/query-provider"
import { ErrorBoundary } from "@/components/error-boundary"
import { Toaster } from "@/components/providers/toast-provider"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary>
          <QueryProvider>
            <div className="relative min-h-screen flex flex-col">
              <Header />
              <main className="flex-1 flex flex-col">
                <div className="container py-6 flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                  {children}
                </div>
              </main>
            </div>
            <Toaster />
          </QueryProvider>
        </ErrorBoundary>
      </body>
    </html>
  )
}