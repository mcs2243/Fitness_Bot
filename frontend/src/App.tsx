import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import { LayoutDashboard, MessageSquare, Dumbbell } from 'lucide-react';
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
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans flex flex-col">
      {/* Top Navigation Bar */}
      <nav className="bg-gray-900/80 backdrop-blur-md border-b border-gray-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="bg-blue-600 p-1.5 rounded-lg">
                <Dumbbell size={20} className="text-white" />
              </div>
              <span className="font-bold text-xl tracking-tight text-white">FitBot</span>
            </div>

            <div className="flex space-x-4">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2
                  ${activeTab === 'dashboard'
                    ? 'bg-gray-800 text-white shadow-sm ring-1 ring-gray-700'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                  }`}
              >
                <LayoutDashboard size={16} />
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('chat')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2
                  ${activeTab === 'chat'
                    ? 'bg-gray-800 text-white shadow-sm ring-1 ring-gray-700'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                  }`}
              >
                <MessageSquare size={16} />
                Coach Chat
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent">
          {activeTab === 'dashboard' ? (
            <Dashboard data={data} onUpload={handleUpload} />
          ) : (
            <div className="h-full max-w-5xl mx-auto p-4 lg:p-8">
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
