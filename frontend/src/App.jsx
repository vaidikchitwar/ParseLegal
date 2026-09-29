import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [dbReady, setDbReady] = useState(false);
  const [docName, setDocName] = useState('');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/api/status')
      .then((res) => res.json())
      .then((data) => {
        setDbReady(data.db_ready);
        if (data.doc_name) setDocName(data.doc_name);
      })
      .catch((err) => console.error(err));
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/ingest', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setDbReady(true);
        setDocName(file.name);
        setMessages([]);
        alert(data.message);
      } else {
        alert('Error: ' + data.detail);
      }
    } catch (err) {
      alert('Error uploading file');
    }
    setLoading(false);
  };

  const loadExistingDb = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/load_existing', {
        method: 'POST',
      });
      const data = await res.json();
      if (res.ok) {
        setDbReady(true);
        setDocName('legal_doc.pdf (pre-loaded)');
        setMessages([]);
        alert(data.message);
      } else {
        alert('Error: ' + data.detail);
      }
    } catch (err) {
      alert('Error loading existing DB');
    }
    setLoading(false);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.answer, sources: data.sources },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Error: ' + data.detail },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error communicating with server' },
      ]);
    }
    setLoading(false);
  };

  const suggestions = [
    "What are the key obligations in this document?",
    "What penalties or consequences are mentioned?",
    "What rights does this document grant?",
    "Are there any dispute resolution clauses?",
    "What Indian law applies to this agreement?",
    "What are the termination conditions?",
  ];

  return (
    <div className="app-container">
      <div className="sidebar">
        <h1 className="app-header">Parse<span className="app-header-accent">Legal</span></h1>
        <p className="app-subtitle">AI-Powered Legal Document Analysis</p>
        <hr />
        
        <h3>Document Source</h3>
        <div className="upload-section">
          <label className="upload-btn">
            Upload New Document
            <input type="file" hidden accept=".pdf,.txt,.md,.docx" onChange={handleFileUpload} />
          </label>
          <button className="secondary-btn" onClick={loadExistingDb}>Load Existing DB</button>
        </div>
        
        <hr />
        <div className={`status-badge ${dbReady ? 'status-indexed' : 'status-waiting'}`}>
          {dbReady ? (
            <>
              <span className="status-dot"></span> Status: Indexed
            </>
          ) : (
            '○ No document loaded'
          )}
        </div>
        {dbReady && <p className="doc-name"><strong>Document:</strong> {docName}</p>}
        
        <hr />
        <h3>Suggested Queries</h3>
        <div className="suggestions">
          {suggestions.map((s, i) => (
            <button key={i} className="suggestion-btn" onClick={() => setInput(s)}>
              {s}
            </button>
          ))}
        </div>
        
        <hr />
        <button className="secondary-btn" onClick={() => setMessages([])}>Clear Chat</button>
      </div>

      <div className="main-content">
        <h2 className="main-title">Parse<span className="main-title-accent">Legal</span></h2>
        <p className="main-sub">Indian Legal Document Assistant</p>

        {!dbReady ? (
          <div className="info-box">
            <p><strong>Get started</strong> — Upload any legal document (PDF, DOCX, TXT) in the sidebar, or load the pre-ingested document. Then ask any question and ParseLegal will answer with reference to Indian law.</p>
          </div>
        ) : (
          <div className="chat-container">
            <div className="messages">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                  {msg.sources && msg.sources.length > 0 && (
                    <details className="sources-expander">
                      <summary>Source References</summary>
                      {msg.sources.map((src, idx) => (
                        <div key={idx} className="source-card">
                          <div className="source-label">
                            <span className="citation-badge">Page {src.page}</span> Excerpt {idx + 1}
                          </div>
                          {src.content.substring(0, 320)}...
                        </div>
                      ))}
                    </details>
                  )}
                </div>
              ))}
              {loading && <div className="loading">Analysing document...</div>}
            </div>

            <div className="chat-input-container">
              <input
                type="text"
                placeholder="Ask a legal question about your document..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              />
              <button onClick={handleSend} disabled={loading}>Send</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
