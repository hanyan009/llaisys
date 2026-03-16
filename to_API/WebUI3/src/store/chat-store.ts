import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { v4 as uuidv4 } from 'uuid'
import localforage from 'localforage'
import { Message, Session } from '@/types'

// Configure localforage
localforage.config({
  name: 'llaisys-chat-storage',
  storeName: 'chats'
});

interface ChatState {
  sessions: Session[];
  currentSessionId: string | null;
  isGenerating: boolean;
  
  // Actions
  createSession: (title?: string) => string;
  deleteSession: (id: string) => void;
  selectSession: (id: string) => void;
  updateSession: (id: string, updates: Partial<Session>) => void;
  
  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'timestamp'> & { id?: string }) => void;
  updateMessage: (sessionId: string, messageId: string, content: string) => void;
  setGenerating: (isGenerating: boolean) => void;
  clearSessions: () => void;
}

const storage = {
  getItem: async (name: string): Promise<any> => {
    console.log(name, "has been retrieved");
    return await localforage.getItem(name);
  },
  setItem: async (name: string, value: any): Promise<void> => {
    console.log(name, "with value", value, "has been saved");
    await localforage.setItem(name, value);
  },
  removeItem: async (name: string): Promise<void> => {
    console.log(name, "has been deleted");
    await localforage.removeItem(name);
  },
};

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      sessions: [],
      currentSessionId: null,
      isGenerating: false,

      createSession: (title) => {
        const id = uuidv4();
        const newSession: Session = {
          id,
          title: title || 'New Chat',
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
          model: 'deepseek-r1',
          providerId: 'local',
          settings: {
            temperature: 0.7,
            topP: 1.0,
            topK: 50,
            argmax: false,
            maxTokens: 2000
          }
        };
        
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSessionId: id
        }));
        
        return id;
      },

      deleteSession: (id) => {
        set((state) => {
          const newSessions = state.sessions.filter(s => s.id !== id);
          let newCurrentId = state.currentSessionId;
          
          if (state.currentSessionId === id) {
            newCurrentId = newSessions.length > 0 ? newSessions[0].id : null;
          }
          
          return {
            sessions: newSessions,
            currentSessionId: newCurrentId
          };
        });
      },

      selectSession: (id) => {
        set({ currentSessionId: id });
      },

      updateSession: (id, updates) => {
        set((state) => ({
          sessions: state.sessions.map(s => 
            s.id === id ? { ...s, ...updates, updatedAt: Date.now() } : s
          )
        }));
      },

      addMessage: (sessionId: string, message: Omit<Message, 'id' | 'timestamp'> & { id?: string }) => {
        const newMessage: Message = {
          id: message.id || uuidv4(),
          timestamp: Date.now(),
          ...message
        };

        set((state) => ({
          sessions: state.sessions.map(s => {
            if (s.id === sessionId) {
              return {
                ...s,
                messages: [...s.messages, newMessage],
                updatedAt: Date.now()
              };
            }
            return s;
          })
        }));
      },

      updateMessage: (sessionId, messageId, content) => {
        set((state) => ({
          sessions: state.sessions.map(s => {
            if (s.id === sessionId) {
              return {
                ...s,
                messages: s.messages.map(m => 
                  m.id === messageId ? { ...m, content } : m
                )
              };
            }
            return s;
          })
        }));
      },

      setGenerating: (isGenerating) => set({ isGenerating }),

      clearSessions: () => set({ sessions: [], currentSessionId: null }),
    }),
    {
      name: 'chat-storage',
      storage: createJSONStorage(() => storage),
    }
  )
)
