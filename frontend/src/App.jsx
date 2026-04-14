import { useEffect, useMemo, useRef, useState } from 'react';

const API_FALLBACK_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function apiFetch(path, options) {
  const normalizedPath = path.startsWith('/api/') ? path : `/api/${path.replace(/^\/+/, '')}`;

  try {
    const response = await fetch(normalizedPath, options);
    if (response.status !== 404 && response.status !== 405) {
      return response;
    }
  } catch (error) {
    // Fall through to direct backend request when proxy/relative routing is unavailable.
  }

  return fetch(`${API_FALLBACK_BASE}${normalizedPath}`, options);
}

function formatConfidence(value) {
  if (typeof value !== 'number') return '-';
  return `${(value * 100).toFixed(1)}%`;
}

export default function App() {
  const nextMessageId = useRef(2);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [corpus, setCorpus] = useState(null);
  const [corpusError, setCorpusError] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      text: 'Ask any question. I will run full SR-RAG routing automatically (SKIP/LITE/FULL, retrieval, screening, optional refuter loop, and judging).',
    },
  ]);
  const [selectedMessageId, setSelectedMessageId] = useState(1);
  const [logs, setLogs] = useState([]);
  const [logLoading, setLogLoading] = useState(false);

  const createMessage = (message) => {
    const id = nextMessageId.current;
    nextMessageId.current += 1;
    return { id, ...message };
  };

  const selectedInsightsMessage = useMemo(() => {
    const selectedIndex = messages.findIndex((message) => message.id === selectedMessageId);
    const fallbackIndex = selectedIndex >= 0 ? selectedIndex : messages.length - 1;

    for (let i = fallbackIndex; i < messages.length; i += 1) {
      if (messages[i].role === 'assistant' && messages[i].payload) return messages[i];
    }

    for (let i = fallbackIndex; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant' && messages[i].payload) return messages[i];
    }

    return null;
  }, [messages, selectedMessageId]);

  useEffect(() => {
    refreshCorpus();
    refreshLogs();
  }, []);

  async function refreshCorpus() {
    try {
      const response = await apiFetch('/api/corpus');
      if (!response.ok) {
        throw new Error('Failed to load corpus state');
      }
      const data = await response.json();
      setCorpus(data.corpus || null);
      setCorpusError('');
    } catch (error) {
      setCorpusError(error.message);
    }
  }

  async function refreshLogs() {
    setLogLoading(true);
    try {
      const response = await apiFetch('/api/logs?lines=120');
      if (!response.ok) {
        throw new Error('Failed to load logs');
      }
      const data = await response.json();
      setLogs(data.lines || []);
    } catch (error) {
      setLogs([`Error loading logs: ${error.message}`]);
    } finally {
      setLogLoading(false);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const userMessage = createMessage({ role: 'user', text });
    setMessages((prev) => [...prev, userMessage]);
    setSelectedMessageId(userMessage.id);
    setInput('');
    setLoading(true);

    try {
      const response = await apiFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(err || 'Request failed');
      }

      const data = await response.json();
      const assistantMessage = createMessage({
        role: 'assistant',
        text: data.answer,
        payload: data,
        payloadJson: JSON.stringify(data, null, 2),
      });
      setMessages((prev) => [
        ...prev,
        assistantMessage,
      ]);
      setSelectedMessageId(assistantMessage.id);
    } catch (error) {
      const assistantErrorMessage = createMessage({
        role: 'assistant',
        text: `Error: ${error.message}`,
      });
      setMessages((prev) => [
        ...prev,
        assistantErrorMessage,
      ]);
      setSelectedMessageId(assistantErrorMessage.id);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || uploading) return;

    setUploading(true);
    try {
      const content = await file.text();
      const response = await apiFetch('/api/corpus/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          content,
          replace: true,
          chunk_size: 1200,
          overlap: 150,
        }),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(err || 'Upload failed');
      }

      const data = await response.json();
      setCorpus(data.corpus || null);
      setCorpusError('');
      const indexedChunks = data?.corpus?.chunk_count ?? 'an unknown number of';
      const source = data?.corpus?.source || 'upload';
      const uploadMessage = createMessage({
        role: 'assistant',
        text: `Corpus replaced with ${file.name}. Indexed ${indexedChunks} chunks into FAISS (source: ${source}).`,
      });
      setMessages((prev) => [
        ...prev,
        uploadMessage,
      ]);
      setSelectedMessageId(uploadMessage.id);
      await refreshCorpus();
      await refreshLogs();
    } catch (error) {
      setCorpusError(error.message);
      const uploadErrorMessage = createMessage({
        role: 'assistant',
        text: `Upload error: ${error.message}`,
      });
      setMessages((prev) => [
        ...prev,
        uploadErrorMessage,
      ]);
      setSelectedMessageId(uploadErrorMessage.id);
    } finally {
      setUploading(false);
    }
  }

  const metadata = selectedInsightsMessage?.payload?.metadata || {};
  const pipelineTrace = Array.isArray(metadata.pipeline_trace) ? metadata.pipeline_trace : [];
  const claimExplanations = Array.isArray(metadata.claim_explanations) ? metadata.claim_explanations : [];
  const loopExecuted = metadata.loop_executed;
  const loopReason = metadata.loop_reason || metadata.abstention_reason || metadata.retrieval_reason || 'No additional explanation available.';

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>SR-RAG Interactive Chat</h1>
        <p>Evidence-grounded answers with claim-level verification.</p>
        <div className="corpus-strip">
          <div>
            <span>Active corpus</span>
            <strong>{corpus?.name || 'Loading corpus...'}</strong>
          </div>
          <div>
            <span>Chunks</span>
            <strong>{corpus?.chunk_count ?? '-'}</strong>
          </div>
          <label className="upload-button">
            <input type="file" accept=".txt,.md,.json,.csv,.jsonl" onChange={handleUpload} />
            {uploading ? 'Uploading...' : 'Upload document'}
          </label>
        </div>
        {corpusError ? <p className="error-line">{corpusError}</p> : null}
        <div className="topbar-actions">
          <button type="button" onClick={refreshLogs} disabled={logLoading}>
            {logLoading ? 'Loading logs...' : 'Refresh logs'}
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="chat-panel">
          <div className="message-list">
            {messages.map((msg, idx) => (
              <article
                key={msg.id || idx}
                className={`message ${msg.role} ${msg.id === selectedMessageId ? 'selected' : ''}`}
                onClick={() => setSelectedMessageId(msg.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    setSelectedMessageId(msg.id);
                  }
                }}
              >
                <span className="role">{msg.role === 'user' ? 'You' : 'SR-RAG'}</span>
                <p>{msg.text}</p>
              </article>
            ))}
          </div>

          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question..."
              rows={3}
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Running pipeline...' : 'Ask'}
            </button>
          </form>
        </section>

        <aside className="insights-panel">
          <h2>Pipeline Insights</h2>
          {!selectedInsightsMessage ? (
            <p className="placeholder">Send a question to see route, loop usage, and claim table.</p>
          ) : (
            <>
              <p className="placeholder">Showing insights for selected message #{selectedMessageId}</p>
              <div className="metric-grid">
                <div className="metric">
                  <span>Route</span>
                  <strong>{selectedInsightsMessage.payload.metadata?.route || '-'}</strong>
                </div>
                <div className="metric">
                  <span>Confidence</span>
                  <strong>{formatConfidence(selectedInsightsMessage.payload.overall_confidence)}</strong>
                </div>
                <div className="metric">
                  <span>Refuter Queue</span>
                  <strong>{selectedInsightsMessage.payload.metadata?.refuter_queue_size ?? 0}</strong>
                </div>
                <div className="metric">
                  <span>Bypass Queue</span>
                  <strong>{selectedInsightsMessage.payload.metadata?.bypass_queue_size ?? 0}</strong>
                </div>
              </div>

              <h3>Why This Route</h3>
              <div className="reason-stack">
                <div className="reason-card">
                  <span>Route reason</span>
                  <p>{metadata.route_reason || 'No route explanation available.'}</p>
                </div>
                <div className="reason-card">
                  <span>Retrieval reason</span>
                  <p>{metadata.retrieval_reason || 'No retrieval explanation available.'}</p>
                </div>
                <div className="reason-card">
                  <span>Loop status</span>
                  <p>{loopExecuted === true ? 'Refuter loop ran.' : loopExecuted === false ? 'Refuter loop did not run.' : 'Loop status unavailable.'}</p>
                  <p>{loopReason}</p>
                </div>
                <div className="reason-card">
                  <span>Abstention reason</span>
                  <p>{metadata.abstention_reason || 'No abstention triggered.'}</p>
                </div>
              </div>

              <h3>Pipeline Trace</h3>
              {pipelineTrace.length ? (
                <ul className="trace-list">
                  {pipelineTrace.map((entry, idx) => {
                    const stage = typeof entry === 'string' ? `Step ${idx + 1}` : entry.stage || entry.name || `Step ${idx + 1}`;
                    const message = typeof entry === 'string' ? entry : entry.message || entry.reason || JSON.stringify(entry);
                    return (
                      <li key={idx}>
                        <strong>{stage}</strong>
                        <p>{message}</p>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="placeholder">No pipeline trace returned for this answer.</p>
              )}

              <h3>Claim Explanations</h3>
              {claimExplanations.length ? (
                <div className="claim-explanations">
                  {claimExplanations.map((entry, idx) => (
                    <article key={idx} className="reason-card">
                      <span>{entry.claim || `Claim ${idx + 1}`}</span>
                      <p>{entry.reason || entry.explanation || entry.verdict || 'No explanation provided.'}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="placeholder">No claim-level explanations were returned.</p>
              )}

              <h3>Claim Table</h3>
              {selectedInsightsMessage.payload.claim_table?.length ? (
                <div className="claim-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Claim</th>
                        <th>Verdict</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedInsightsMessage.payload.claim_table.map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.claim}</td>
                          <td>{row.verdict}</td>
                          <td>{row.confidence}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="placeholder">No claim table for this response.</p>
              )}

              <h3>Raw JSON</h3>
              <div className="log-box">
                <pre>{selectedInsightsMessage.payloadJson || JSON.stringify(selectedInsightsMessage.payload || {}, null, 2)}</pre>
              </div>

              <h3>Recent Logs</h3>
              <div className="log-box">
                {logs.length ? (
                  <pre>{logs.join('\n')}</pre>
                ) : (
                  <p className="placeholder">Click Refresh logs to load recent backend output, including HF fetch attempts.</p>
                )}
              </div>
            </>
          )}
        </aside>
      </main>
    </div>
  );
}
