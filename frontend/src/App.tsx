import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import axios from 'axios';
import {
  TabList,
  Tab,
  makeStyles,
  shorthands,
  Text,
  Avatar
} from '@fluentui/react-components';
import {
  Board24Regular,
  Chat24Regular,
  Dumbbell24Regular
} from '@fluentui/react-icons';

// Configure axios base URL
axios.defaults.baseURL = 'http://localhost:8000';

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: '#202020', // Fluent dark background
    color: '#fff',
  },
  nav: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 24px',
    backgroundColor: '#292929',
    borderBottom: '1px solid #424242',
    height: '60px',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  content: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  scrollContainer: {
    flex: 1,
    overflowY: 'auto',
    ...shorthands.padding('24px'),
  }
});

function App() {
  const styles = useStyles();
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
    <div className={styles.root}>
      <nav className={styles.nav}>
        <div className={styles.logo}>
          <Avatar icon={<Dumbbell24Regular />} color="brand" />
          <Text weight="semibold" size={400}>FitBot</Text>
        </div>

        <TabList
          selectedValue={activeTab}
          onTabSelect={(_, data) => setActiveTab(data.value as 'dashboard' | 'chat')}
        >
          <Tab value="dashboard" icon={<Board24Regular />}>Dashboard</Tab>
          <Tab value="chat" icon={<Chat24Regular />}>Coach Chat</Tab>
        </TabList>

        <div style={{ width: '40px' }}></div> {/* Spacer for balance */}
      </nav>

      <main className={styles.content}>
        <div className={styles.scrollContainer}>
          {activeTab === 'dashboard' ? (
            <Dashboard data={data} onUpload={handleUpload} />
          ) : (
            <div style={{ maxWidth: '1000px', margin: '0 auto', height: '100%' }}>
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
