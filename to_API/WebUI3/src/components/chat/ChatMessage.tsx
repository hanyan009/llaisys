import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import { cn } from '@/lib/utils'
import { Message } from '@/types'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { User, Bot, Copy, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ChatMessageProps {
  message: Message
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user'

  return (
    <div
      className={cn(
        "group flex w-full gap-4 p-4 md:px-8",
        isUser ? "bg-background" : "bg-muted/30"
      )}
    >
      <Avatar className={cn("h-8 w-8", isUser ? "order-2" : "order-1")}>
        <AvatarFallback className={isUser ? "bg-primary text-primary-foreground" : "bg-muted-foreground text-background"}>
          {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
        </AvatarFallback>
      </Avatar>

      <div className={cn("flex-1 space-y-2 overflow-hidden", isUser ? "order-1 text-right" : "order-2")}>
        <div className={cn("prose dark:prose-invert max-w-none break-words", isUser && "ml-auto")}>
          {isUser ? (
             <div className="bg-primary text-primary-foreground px-4 py-2 rounded-lg inline-block text-left">
                {message.content}
             </div>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeHighlight, rehypeKatex]}
              components={{
                pre: ({ node, ...props }) => (
                  <div className="relative my-4 overflow-hidden rounded-lg border bg-muted p-0">
                    <div className="flex items-center justify-between bg-muted px-4 py-2 text-xs text-muted-foreground">
                      <span>Code</span>
                      <CopyButton content={String((node as any)?.children?.[0]?.children?.[0]?.value || '')} />
                    </div>
                    <pre {...props} className="overflow-x-auto p-4" />
                  </div>
                ),
                code: ({ node, className, children, ...props }) => {
                  const match = /language-(\w+)/.exec(className || '')
                  return match ? (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  ) : (
                    <code className="rounded bg-muted px-1 py-0.5 font-mono text-sm" {...props}>
                      {children}
                    </code>
                  )
                }
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>
        {message.isError && (
          <div className="text-sm text-destructive mt-2">
            Error sending message.
          </div>
        )}
      </div>
    </div>
  )
}

const CopyButton = ({ content }: { content: string }) => {
  const [copied, setCopied] = useState(false)

  const onCopy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-6 w-6 text-muted-foreground hover:text-foreground"
      onClick={onCopy}
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </Button>
  )
}
