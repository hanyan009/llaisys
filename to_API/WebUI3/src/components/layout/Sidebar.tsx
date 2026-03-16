import React from 'react'
import { useChatStore } from '@/store/chat-store'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Plus, MessageSquare, Trash2, Settings, MoreHorizontal } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { formatDistanceToNow } from 'date-fns'

interface SidebarProps {
  className?: string
  onSettingsClick: () => void
}

export const Sidebar: React.FC<SidebarProps> = ({ className, onSettingsClick }) => {
  const { sessions, currentSessionId, createSession, selectSession, deleteSession } = useChatStore()

  const handleNewChat = () => {
    createSession()
  }

  return (
    <div className={cn("flex flex-col h-full border-r bg-muted/10", className)}>
      <div className="p-4 border-b">
        <Button onClick={handleNewChat} className="w-full justify-start gap-2" variant="outline">
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2 space-y-2">
          {sessions.length === 0 && (
            <div className="text-center text-sm text-muted-foreground py-8">
              No chats yet.
            </div>
          )}
          {sessions.map((session) => (
            <div
              key={session.id}
              className={cn(
                "group flex items-center justify-between rounded-md px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground cursor-pointer transition-colors",
                currentSessionId === session.id ? "bg-accent text-accent-foreground" : "text-muted-foreground"
              )}
              onClick={() => selectSession(session.id)}
            >
              <div className="flex items-center gap-3 overflow-hidden">
                <MessageSquare className="h-4 w-4 shrink-0" />
                <div className="flex flex-col overflow-hidden text-left">
                    <span className="truncate">{session.title || 'New Chat'}</span>
                    <span className="text-[10px] text-muted-foreground/60 font-normal">
                        {formatDistanceToNow(session.updatedAt, { addSuffix: true })}
                    </span>
                </div>
              </div>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <MoreHorizontal className="h-3 w-3" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem 
                    className="text-destructive focus:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteSession(session.id)
                    }}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="p-4 border-t">
        <Button variant="ghost" className="w-full justify-start gap-2" onClick={onSettingsClick}>
          <Settings className="h-4 w-4" />
          Settings
        </Button>
      </div>
    </div>
  )
}
