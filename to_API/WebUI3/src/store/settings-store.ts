import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { AppSettings, Provider } from '@/types'
import { v4 as uuidv4 } from 'uuid'

interface SettingsState {
  settings: AppSettings;
  providers: Provider[];
  
  // Actions
  setTheme: (theme: AppSettings['theme']) => void;
  setFontSize: (size: AppSettings['fontSize']) => void;
  
  addProvider: (provider: Omit<Provider, 'id'>) => void;
  updateProvider: (id: string, updates: Partial<Provider>) => void;
  removeProvider: (id: string) => void;
  
  setDefaultModel: (providerId: string, model: string) => void;
}

const LOCAL_PROVIDER_DEFAULT: Provider = {
  id: 'local',
  name: 'Local/Custom',
  baseUrl: 'http://localhost:8812/v1',
  apiKey: 'sk-no-key-required',
  enabled: true,
  models: ['deepseek-r1']
}

const applyDefaultLocalProvider = (providers: Provider[]): Provider[] => {
  const hasLocal = providers.some((provider) => provider.id === 'local')
  if (!hasLocal) {
    return [...providers, LOCAL_PROVIDER_DEFAULT]
  }

  return providers.map((provider) =>
    provider.id === 'local'
      ? {
          ...provider,
          baseUrl: LOCAL_PROVIDER_DEFAULT.baseUrl,
          enabled: true,
          models: provider.models.includes('deepseek-r1')
            ? provider.models
            : ['deepseek-r1', ...provider.models]
        }
      : provider
  )
}

const DEFAULT_PROVIDERS: Provider[] = [
  {
    id: 'openai',
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    apiKey: '',
    enabled: true,
    models: ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4-turbo']
  },
  {
    ...LOCAL_PROVIDER_DEFAULT
  }
];

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      settings: {
        theme: 'system',
        fontSize: 'medium',
        defaultProviderId: 'local',
        defaultModel: 'deepseek-r1',
        defaultTemperature: 0.7,
        defaultTopP: 1.0,
        defaultMaxTokens: 2000,
      },
      providers: DEFAULT_PROVIDERS,

      setTheme: (theme) => set((state) => ({
        settings: { ...state.settings, theme }
      })),
      
      setFontSize: (size) => set((state) => ({
        settings: { ...state.settings, fontSize: size }
      })),

      addProvider: (provider) => set((state) => ({
        providers: [...state.providers, { ...provider, id: uuidv4() }]
      })),

      updateProvider: (id, updates) => set((state) => ({
        providers: state.providers.map((p) => 
          p.id === id ? { ...p, ...updates } : p
        )
      })),

      removeProvider: (id) => set((state) => ({
        providers: state.providers.filter((p) => p.id !== id)
      })),

      setDefaultModel: (providerId, model) => set((state) => ({
        settings: { 
          ...state.settings, 
          defaultProviderId: providerId, 
          defaultModel: model 
        }
      })),
    }),
    {
      name: 'llaisys-settings',
      version: 3,
      migrate: (persistedState) => {
        const state = persistedState as Partial<SettingsState> | undefined
        const providers = applyDefaultLocalProvider(state?.providers || DEFAULT_PROVIDERS)
        return {
          ...state,
          providers,
          settings: {
            theme: state?.settings?.theme || 'system',
            fontSize: state?.settings?.fontSize || 'medium',
            defaultProviderId: 'local',
            defaultModel: 'deepseek-r1',
            defaultTemperature: state?.settings?.defaultTemperature ?? 0.7,
            defaultTopP: state?.settings?.defaultTopP ?? 1.0,
            defaultMaxTokens: state?.settings?.defaultMaxTokens ?? 2000,
          },
        }
      },
    }
  )
)
