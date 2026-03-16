import React, { useState } from 'react'
import { Sidebar } from './Sidebar'
import { SettingsDialog } from '@/components/settings/SettingsDialog'
import { Button } from '@/components/ui/button'
import { Menu, X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LayoutProps {
  children: React.ReactNode
  isSettingsOpen: boolean
  onSettingsOpenChange: (open: boolean) => void
}

export const Layout: React.FC<LayoutProps> = ({ children, isSettingsOpen, onSettingsOpenChange }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  // Close sidebar on route change or when needed (not applicable here as SPA)
  // Close sidebar when clicking outside on mobile
  
  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={cn(
        "fixed inset-y-0 left-0 z-50 w-72 transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0 bg-background border-r",
        isSidebarOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between p-4 border-b md:hidden">
            <span className="font-semibold">Menu</span>
            <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(false)}>
              <X className="h-5 w-5" />
            </Button>
          </div>
          <Sidebar 
            className="flex-1 border-r-0" 
            onSettingsClick={() => {
              onSettingsOpenChange(true)
              setIsSidebarOpen(false)
            }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 flex-col overflow-hidden relative">
        <header className="flex items-center border-b p-4 md:hidden">
          <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(true)}>
            <Menu className="h-5 w-5" />
          </Button>
          <span className="ml-4 font-semibold">LLAISYS WebUI</span>
        </header>
        <main className="flex-1 overflow-hidden relative flex flex-col">
          {children}
        </main>
      </div>

      <SettingsDialog open={isSettingsOpen} onOpenChange={onSettingsOpenChange} />
    </div>
  )
}
