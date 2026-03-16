import { useEffect, useState } from 'react'
import { Layout } from '@/components/layout/Layout'
import { ChatList } from '@/components/chat/ChatList'
import { ChatInput } from '@/components/chat/ChatInput'
import { ChatHeader } from '@/components/chat/ChatHeader'
import { useChatStore } from '@/store/chat-store'
import { useSettingsStore } from '@/store/settings-store'
import { chatCompletion } from '@/lib/api'
import { Message } from '@/types'

const resolveErrorMessage = (error: unknown) => {
  if (error instanceof Error && error.message) return error.message
  if (typeof error === 'string' && error.trim()) return error
  return '请求失败，请检查网络或接口配置'
}

function App() {
  const { 
    sessions, 
    currentSessionId, 
    createSession, 
    addMessage, 
    updateMessage,
    updateSession,
    setGenerating, 
    isGenerating 
  } = useChatStore()
  
  const { settings, providers } = useSettingsStore()
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  // Initialize session if none exists
  useEffect(() => {
    if (sessions.length === 0) {
      createSession()
    } else if (!currentSessionId) {
        // Select the first session if none selected
        useChatStore.getState().selectSession(sessions[0].id)
    }
  }, [sessions.length, createSession, currentSessionId, sessions])

  const currentSession = sessions.find(s => s.id === currentSessionId)
  const messages = currentSession?.messages || []

  const handleSettingsChange = (newSettings: any) => {
    if (currentSessionId) {
      updateSession(currentSessionId, { settings: newSettings })
    }
  }

  const handleSend = async (content: string) => {
    if (!currentSessionId) return

    // 1. Add user message
    const userMessage: Omit<Message, 'id' | 'timestamp'> = {
      role: 'user',
      content
    }
    addMessage(currentSessionId, userMessage)
    
    // 2. Add placeholder assistant message
    const assistantMessageId = crypto.randomUUID()
    addMessage(currentSessionId, {
      id: assistantMessageId,
      role: 'assistant',
      content: ''
    })

    // 3. Prepare for API call
    setGenerating(true)
    const controller = new AbortController()
    setAbortController(controller)

    try {
      // Find provider
      const provider = providers.find(p => p.id === (currentSession?.providerId || settings.defaultProviderId)) 
        || providers.find(p => p.enabled) 
        || providers[0]
      
      if (!provider) throw new Error('No provider available')

      const model = currentSession?.model || settings.defaultModel || 'gpt-3.5-turbo'
      
      // Call API
      let fullResponse = ''
      await chatCompletion(
        [...messages, { ...userMessage, id: 'temp', timestamp: Date.now() } as Message],
        provider,
        model,
        {
          temperature: currentSession?.settings?.temperature ?? settings.defaultTemperature,
          topP: currentSession?.settings?.topP ?? settings.defaultTopP,
          topK: currentSession?.settings?.topK ?? 50,
          argmax: currentSession?.settings?.argmax ?? false,
          maxTokens: currentSession?.settings?.maxTokens ?? settings.defaultMaxTokens
        },
        (chunk) => {
          if (controller.signal.aborted) return
          fullResponse += chunk
          updateMessage(currentSessionId, assistantMessageId, fullResponse)
        },
        controller.signal
      )
      
    } catch (error: unknown) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        const errorMessage = resolveErrorMessage(error)
        console.error('Chat error:', error)
        updateMessage(currentSessionId, assistantMessageId, `错误：${errorMessage}`)
      }
    } finally {
      setGenerating(false)
      setAbortController(null)
    }
  }

  const handleStop = () => {
    if (abortController) {
      abortController.abort()
      setAbortController(null)
      setGenerating(false)
    }
  }

  return (
    <Layout isSettingsOpen={isSettingsOpen} onSettingsOpenChange={setIsSettingsOpen}>
      <div className="flex flex-col h-full relative">
        <ChatHeader onOpenSettings={() => setIsSettingsOpen(true)} />
        <ChatList messages={messages} />
        <ChatInput 
          onSend={handleSend} 
          onStop={handleStop} 
          isGenerating={isGenerating}
          settings={currentSession?.settings || {
            temperature: settings.defaultTemperature,
            topP: settings.defaultTopP,
            topK: 50,
            argmax: false,
            maxTokens: settings.defaultMaxTokens
          }}
          onSettingsChange={handleSettingsChange}
        />
      </div>
    </Layout>
  )
}

export default App
