import React from 'react'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Settings2 } from 'lucide-react'

interface ChatSettingsProps {
  settings: {
    temperature: number
    topP: number
    topK?: number
    argmax?: boolean
    maxTokens: number
  }
  onSettingsChange: (settings: any) => void
}

export const ChatSettings: React.FC<ChatSettingsProps> = ({ settings, onSettingsChange }) => {
  const handleChange = (key: string, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    })
  }

  const isArgmax = settings.argmax || false

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full">
          <Settings2 className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="start" side="top">
        <div className="grid gap-4">
          <div className="space-y-2">
            <h4 className="font-medium leading-none">Generation Settings</h4>
            <p className="text-sm text-muted-foreground">
              Adjust parameters for the current session.
            </p>
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="argmax">Argmax (Greedy)</Label>
            <Switch
              id="argmax"
              checked={isArgmax}
              onCheckedChange={(checked) => handleChange('argmax', checked)}
            />
          </div>

          {!isArgmax && (
            <>
              <div className="grid gap-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="temperature">Temperature</Label>
                  <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground hover:border-border">
                    {settings.temperature}
                  </span>
                </div>
                <Slider
                  id="temperature"
                  max={2}
                  step={0.1}
                  value={[settings.temperature]}
                  onValueChange={(value) => handleChange('temperature', value[0])}
                />
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="topP">Top P</Label>
                  <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground hover:border-border">
                    {settings.topP}
                  </span>
                </div>
                <Slider
                  id="topP"
                  max={1}
                  step={0.05}
                  value={[settings.topP]}
                  onValueChange={(value) => handleChange('topP', value[0])}
                />
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="topK">Top K</Label>
                  <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground hover:border-border">
                    {settings.topK || 50}
                  </span>
                </div>
                <Slider
                  id="topK"
                  max={100}
                  step={1}
                  value={[settings.topK || 50]}
                  onValueChange={(value) => handleChange('topK', value[0])}
                />
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="maxTokens">Max Tokens</Label>
                  <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground hover:border-border">
                    {settings.maxTokens || 2000}
                  </span>
                </div>
                <Slider
                  id="maxTokens"
                  max={4096}
                  step={100}
                  value={[settings.maxTokens || 2000]}
                  onValueChange={(value) => handleChange('maxTokens', value[0])}
                />
              </div>
            </>
          )}
          
           {isArgmax && (
             <p className="text-xs text-muted-foreground italic">
               Argmax enabled. Other parameters are disabled.
             </p>
           )}
        </div>
      </PopoverContent>
    </Popover>
  )
}
