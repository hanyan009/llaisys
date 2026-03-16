import React, { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useSettingsStore } from '@/store/settings-store'
import { Provider } from '@/types'
import { Plus, Trash2, Edit2 } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export const SettingsDialog: React.FC<SettingsDialogProps> = ({ open, onOpenChange }) => {
  const [activeTab, setActiveTab] = useState<'general' | 'providers'>('general')
  const { settings, providers, setTheme, setFontSize, addProvider, updateProvider, removeProvider } = useSettingsStore()

  // Provider Form State
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null)
  const [modelsText, setModelsText] = useState('')
  const [newProvider, setNewProvider] = useState<Partial<Provider>>({
    name: '',
    baseUrl: '',
    apiKey: '',
    enabled: true,
    models: []
  })

  const parseModels = (value: string) => Array.from(
    new Set(value.split(',').map((model) => model.trim()).filter(Boolean))
  )

  const handleSaveProvider = () => {
    const parsedModels = parseModels(modelsText)
    const providerPayload = {
      ...newProvider,
      models: parsedModels
    }

    if (editingProviderId === 'new') {
      // Add new
      if (newProvider.name && newProvider.baseUrl) {
        addProvider(providerPayload as Omit<Provider, 'id'>)
      }
    } else if (editingProviderId) {
      // Update existing
      updateProvider(editingProviderId, providerPayload)
    }
    setEditingProviderId(null)
    setModelsText('')
    setNewProvider({ name: '', baseUrl: '', apiKey: '', enabled: true, models: [] })
  }

  const startEditProvider = (provider: Provider) => {
    setEditingProviderId(provider.id)
    setNewProvider(provider)
    setModelsText(provider.models.join(', '))
  }

  const startNewProvider = () => {
    setEditingProviderId('new')
    setNewProvider({ name: '', baseUrl: '', apiKey: '', enabled: true, models: [] })
    setModelsText('')
  }

  const cancelEditProvider = () => {
    setEditingProviderId(null)
    setModelsText('')
    setNewProvider({ name: '', baseUrl: '', apiKey: '', enabled: true, models: [] })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl h-[80vh] flex flex-col p-0 overflow-hidden gap-0">
        <DialogHeader className="px-6 py-4 border-b">
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Manage your preferences and API providers.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-48 border-r bg-muted/30 p-2 space-y-1">
            <Button
              variant={activeTab === 'general' ? 'secondary' : 'ghost'}
              className="w-full justify-start"
              onClick={() => setActiveTab('general')}
            >
              General
            </Button>
            <Button
              variant={activeTab === 'providers' ? 'secondary' : 'ghost'}
              className="w-full justify-start"
              onClick={() => setActiveTab('providers')}
            >
              Providers
            </Button>
          </div>

          {/* Content */}
          <ScrollArea className="flex-1">
             <div className="p-6">
            {activeTab === 'general' && (
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label>Theme</Label>
                  <Select
                    value={settings.theme}
                    onValueChange={(value: any) => setTheme(value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select theme" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Font Size</Label>
                  <Select
                    value={settings.fontSize}
                    onValueChange={(value: any) => setFontSize(value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select font size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="small">Small</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="large">Large</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}

            {activeTab === 'providers' && (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-medium">API Providers</h3>
                  {!editingProviderId && (
                    <Button size="sm" onClick={startNewProvider}>
                      <Plus className="h-4 w-4 mr-2" />
                      Add Provider
                    </Button>
                  )}
                </div>

                {editingProviderId && (
                  <div className="border rounded-md p-4 space-y-4 bg-muted/20">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Name</Label>
                        <Input
                          value={newProvider.name}
                          onChange={(e) => setNewProvider({ ...newProvider, name: e.target.value })}
                          placeholder="My Provider"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Base URL</Label>
                        <Input
                          value={newProvider.baseUrl}
                          onChange={(e) => setNewProvider({ ...newProvider, baseUrl: e.target.value })}
                          placeholder="https://api.openai.com/v1"
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label>API Key</Label>
                      <Input
                        type="password"
                        value={newProvider.apiKey}
                        onChange={(e) => setNewProvider({ ...newProvider, apiKey: e.target.value })}
                        placeholder="sk-..."
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Models</Label>
                      <Input
                        value={modelsText}
                        onChange={(e) => setModelsText(e.target.value)}
                        placeholder="gpt-4o, gpt-4.1, qwen-plus"
                      />
                      <div className="flex flex-wrap gap-1.5">
                        {parseModels(modelsText).map((model) => (
                          <span
                            key={model}
                            className="text-xs px-2 py-1 rounded-full bg-secondary text-secondary-foreground"
                          >
                            {model}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center justify-between rounded-md border px-3 py-2">
                      <Label htmlFor="provider-enabled">Enabled</Label>
                      <Switch
                        id="provider-enabled"
                        checked={newProvider.enabled ?? true}
                        onCheckedChange={(checked) => setNewProvider({ ...newProvider, enabled: checked })}
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="ghost" size="sm" onClick={cancelEditProvider}>Cancel</Button>
                      <Button size="sm" onClick={handleSaveProvider}>Save</Button>
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  {providers.map((provider) => (
                    <div key={provider.id} className="flex items-center justify-between p-3 border rounded-md hover:bg-muted/50">
                      <div>
                        <div className="font-medium flex items-center gap-2">
                          {provider.name}
                          {provider.enabled && <span className="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded-full dark:bg-green-900 dark:text-green-100">Active</span>}
                        </div>
                        <div className="text-xs text-muted-foreground">{provider.baseUrl}</div>
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {(provider.models.length ? provider.models : ['No models']).map((model) => (
                            <span
                              key={model}
                              className="text-[11px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground"
                            >
                              {model}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <Button variant="ghost" size="icon" onClick={() => startEditProvider(provider)}>
                          <Edit2 className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="icon" className="text-destructive hover:text-destructive" onClick={() => removeProvider(provider.id)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
             </div>
          </ScrollArea>
        </div>
      </DialogContent>
    </Dialog>
  )
}
