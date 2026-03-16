import React, { useRef, useEffect, useState } from 'react'
import { Message } from '@/types'
import { ChatMessage } from './ChatMessage'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useChatStore } from '@/store/chat-store'
import { Loader2 } from 'lucide-react'

interface ChatListProps {
  messages: Message[]
}

export const ChatList: React.FC<ChatListProps> = ({ messages }) => {
  const isGenerating = useChatStore((state) => state.isGenerating)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isGenerating) {
      const startTime = Date.now()
      setElapsed(0)
      interval = setInterval(() => {
        setElapsed(Date.now() - startTime)
      }, 100)
    } else {
      setElapsed(0)
    }
    return () => clearInterval(interval)
  }, [isGenerating])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isGenerating])

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="flex flex-col gap-4 pb-12 max-w-4xl mx-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[50vh] text-center text-muted-foreground">
            <h2 className="text-2xl font-bold mb-2">Welcome to LLAISYS AI</h2>
            <p>Start a conversation by typing a message below.</p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        {isGenerating && (
          <div className="flex items-center gap-2 text-muted-foreground px-8">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Thinking ({(elapsed / 1000).toFixed(1)}s)...</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  )
}
