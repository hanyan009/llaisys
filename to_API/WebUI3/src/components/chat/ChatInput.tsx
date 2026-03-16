import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { SendHorizontal, StopCircle } from 'lucide-react'
import { ChatSettings } from './ChatSettings'

interface ChatInputProps {
  onSend: (content: string) => void
  onStop: () => void
  isGenerating: boolean
  settings: {
    temperature: number
    topP: number
    topK?: number
    argmax?: boolean
    maxTokens: number
  }
  onSettingsChange: (settings: any) => void
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, onStop, isGenerating, settings, onSettingsChange }) => {
  const [content, setContent] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSend = () => {
    if (!content.trim() || isGenerating) return
    onSend(content)
    setContent('')
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'inherit'
      const scrollHeight = textareaRef.current.scrollHeight
      textareaRef.current.style.height = `${Math.min(scrollHeight, 200)}px`
    }
  }, [content])

  return (
    <div className="p-4 border-t bg-background">
      <div className="relative max-w-4xl mx-auto">
        <div className="relative flex items-end gap-2 rounded-xl border bg-background shadow-sm ring-offset-background focus-within:ring-1 focus-within:ring-ring">
          <div className="absolute left-2 bottom-2 z-10">
            <ChatSettings settings={settings} onSettingsChange={onSettingsChange} />
          </div>
          <Textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            className="min-h-[60px] max-h-[200px] w-full resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 py-4 pr-12 pl-12"
            disabled={isGenerating}
          />
          <div className="absolute right-2 bottom-2">
            {isGenerating ? (
                <Button size="icon" variant="destructive" onClick={onStop} className="h-8 w-8 rounded-full">
                <StopCircle className="h-4 w-4" />
                </Button>
            ) : (
                <Button 
                size="icon" 
                onClick={handleSend} 
                disabled={!content.trim()} 
                className="h-8 w-8 rounded-full"
                >
                <SendHorizontal className="h-4 w-4" />
                </Button>
            )}
            </div>
        </div>
        <div className="text-center text-xs text-muted-foreground mt-2">
            AI generated content may be inaccurate.
        </div>
      </div>
    </div>
  )
}
