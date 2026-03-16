export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isError?: boolean;
}

export interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  model: string;
  providerId: string;
  settings?: {
    temperature: number;
    topP: number;
    topK: number;
    argmax: boolean;
    maxTokens: number;
  };
}

export interface Provider {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  enabled: boolean;
  models: string[]; // Manually or auto-fetched models
}

export interface AppSettings {
  theme: 'light' | 'dark' | 'system';
  fontSize: 'small' | 'medium' | 'large';
  defaultProviderId?: string;
  defaultModel?: string;
  defaultTemperature: number;
  defaultTopP: number;
  defaultMaxTokens: number;
}
