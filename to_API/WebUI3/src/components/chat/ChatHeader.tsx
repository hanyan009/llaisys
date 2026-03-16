import React from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useChatStore } from '@/store/chat-store'
import { useSettingsStore } from '@/store/settings-store'
import { Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ChatHeaderProps {
  onOpenSettings: () => void
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({ onOpenSettings }) => {
  const { currentSessionId, sessions, updateSession } = useChatStore()
  const { providers, settings } = useSettingsStore()

  const currentSession = sessions.find(s => s.id === currentSessionId)
  
  // Safe defaults
  const currentProviderId = currentSession?.providerId || settings.defaultProviderId || (providers.length > 0 ? providers[0].id : '')
  const currentProvider = providers.find(p => p.id === currentProviderId)
  
  // Use session model, or provider's first model, or default from settings
  const currentModel = currentSession?.model || (currentProvider?.models.length ? currentProvider.models[0] : settings.defaultModel) || 'gpt-3.5-turbo'

  const handleProviderChange = (providerId: string) => {
    if (!currentSession) return
    const provider = providers.find(p => p.id === providerId)
    if (provider) {
        updateSession(currentSession.id, { 
            providerId: provider.id,
            model: provider.models[0] || 'gpt-3.5-turbo'
        })
    }
  }

  const handleModelChange = (model: string) => {
    if (!currentSession) return
    updateSession(currentSession.id, { model })
  }

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b h-14 bg-background/95 backdrop-blur z-10 shrink-0">
        <div className="flex items-center gap-4 flex-1 overflow-hidden">
            <h2 className="text-lg font-semibold truncate max-w-[150px] hidden sm:block">
              {currentSession?.title || 'New Chat'}
            </h2>
            
            <div className="flex items-center gap-2">
                <Select value={currentProviderId} onValueChange={handleProviderChange}>
                    <SelectTrigger className="w-[140px] h-8 text-xs">
                        <SelectValue placeholder="Provider" />
                    </SelectTrigger>
                    <SelectContent>
                        {providers.map(p => (
                            <SelectItem key={p.id} value={p.id}>
                              <div className="flex items-center gap-2">
                                <span>{p.name}</span>
                                {!p.enabled && <span className="text-[10px] text-muted-foreground">(Disabled)</span>}
                              </div>
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>

                <Select value={currentModel} onValueChange={handleModelChange}>
                    <SelectTrigger className="w-[180px] h-8 text-xs">
                        <SelectValue placeholder="Model" />
                    </SelectTrigger>
                    <SelectContent>
                        {currentProvider?.models.map(m => (
                            <SelectItem key={m} value={m}>{m}</SelectItem>
                        )) || <SelectItem value="gpt-3.5-turbo">gpt-3.5-turbo</SelectItem>}
                    </SelectContent>
                </Select>
            </div>
        </div>
        
        <Button variant="ghost" size="icon" onClick={onOpenSettings} title="Settings">
            <Settings className="h-5 w-5" />
        </Button>
    </div>
  )
}
