interface PageHeaderProps {
  heading: string
  text?: string
  children?: React.ReactNode
}

export function PageHeader({ heading, text, children }: PageHeaderProps) {
  return (
    <div className="flex flex-col items-center gap-4 space-y-2 px-4 sm:px-6 lg:px-8 pb-8 pt-6 md:pt-8">
      <div className="flex max-w-[980px] flex-col items-center gap-2 text-center">
        <h1 className="text-3xl font-bold leading-tight tracking-tight md:text-4xl">
          {heading}
        </h1>
        {text && (
          <p className="max-w-[750px] text-lg text-muted-foreground sm:text-xl">
            {text}
          </p>
        )}
      </div>
      {children}
    </div>
  )
}
