import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import { LayoutDashboard, MessageSquare } from 'lucide-react';
import axios from 'axios';

// Configure axios base URL
axios.defaults.baseURL = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'chat'>('dashboard');
  const [data, setData] = useState<any[]>([]);
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'assistant', content: string }[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Load initial data
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const res = await axios.get('/data');
      setData(res.data.records);
    } catch (err) {
      console.error("Failed to fetch data", err);
    }
  };

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    // We let the error propagate so Dashboard can catch it and show error state
    await axios.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    await fetchData(); // Refresh data
  };

  const handleSendMessage = async (message: string) => {
    const newHistory = [...chatHistory, { role: 'user' as const, content: message }];
    setChatHistory(newHistory);
    setIsChatLoading(true);

    try {
      const res = await axios.post('/chat', {
        message,
        history: newHistory
      });
      setChatHistory([...newHistory, { role: 'assistant', content: res.data.response }]);
    } catch (err) {
      console.error("Chat failed", err);
      setChatHistory([...newHistory, { role: 'assistant', content: "Sorry, I encountered an error. Please check your connection or API keys." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex font-sans">
      {/* Sidebar */}
      <aside className="w-20 lg:w-64 bg-gray-900 border-r border-gray-800 flex flex-col items-center lg:items-stretch py-8 transition-all duration-300">
        <div className="mb-12 px-4 text-center lg:text-left">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent hidden lg:block tracking-tight">
            FitBot
          </h1>
          <span className="text-2xl font-bold text-blue-500 lg:hidden">FB</span>
        </div>

        <nav className="flex-1 space-y-2 px-2">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`w-full p-3 rounded-xl flex items-center gap-3 transition-all duration-200 ${activeTab === 'dashboard'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25 translate-x-1'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white hover:translate-x-1'
              }`}
          >
            <LayoutDashboard size={24} />
            <span className="hidden lg:block font-medium">Dashboard</span>
          </button>

          <button
            onClick={() => setActiveTab('chat')}
            className={`w-full p-3 rounded-xl flex items-center gap-3 transition-all duration-200 ${activeTab === 'chat'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25 translate-x-1'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white hover:translate-x-1'
              }`}
          >
            <MessageSquare size={24} />
            <span className="hidden lg:block font-medium">Coach Chat</span>
          </button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-screen overflow-hidden flex flex-col bg-gray-950">
        <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent">
          {activeTab === 'dashboard' ? (
            <Dashboard data={data} onUpload={handleUpload} />
          ) : (
            <div className="h-full p-4 lg:p-8 max-w-5xl mx-auto w-full">
              <ChatInterface
                history={chatHistory}
                onSendMessage={handleSendMessage}
                isLoading={isChatLoading}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
