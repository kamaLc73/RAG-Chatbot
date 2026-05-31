import { useState } from 'react';
import ChatInterface from '../components/chat/ChatInterface';
import Header from '../components/layout/Header';
import Sidebar from '../components/sidebar/Sidebar';

export default function Chat() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      <Header onMenuClick={() => setIsSidebarOpen(true)} />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />
        <main className="min-w-0 flex-1 overflow-hidden">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
}
