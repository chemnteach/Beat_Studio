/**
 * LoRAManager — browse, download, train, and manage LoRAs.
 * Can run as a full tab or embedded inside VideoStudio.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { LoRAEntry, LoRASearchResult } from '../types';

interface Props {
  audioId?: string;
  style?: string;
  embedded?: boolean;
  selectedNames?: string[];
  onSelectionChange?: (names: string[]) => void;
}

export function LoRAManager({ audioId, style, embedded = false, selectedNames = [], onSelectionChange }: Props) {
  const [loras, setLoras] = useState<LoRAEntry[]>([]);
  const [searchResults, setSearchResults] = useState<LoRASearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);

  const toggleLora = (name: string) => {
    if (!onSelectionChange) return;
    const next = selectedNames.includes(name)
      ? selectedNames.filter(n => n !== name)
      : [...selectedNames, name];
    onSelectionChange(next);
  };

  useEffect(() => {
    void loadLoras();
  }, [filterType]);

  const loadLoras = async () => {
    setLoading(true);
    try {
      const params = filterType !== 'all' ? { type_filter: filterType } : {};
      const { data } = await axios.get<{ loras: LoRAEntry[] }>('/api/lora/list', { params });
      setLoras(data.loras);
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  const searchOnline = async () => {
    if (!searchQuery.trim()) return;
    try {
      const { data } = await axios.post<{ downloadable: LoRASearchResult[] }>('/api/lora/recommend', {
        audio_id: audioId ?? '',
        style: style ?? '',
      });
      setSearchResults(data.downloadable ?? []);
    } catch { /* non-fatal */ }
  };

  const downloadLora = async (result: LoRASearchResult) => {
    setDownloading(result.name);
    try {
      await axios.post('/api/lora/download', {
        url: result.url,
        name: result.name,
        lora_type: result.type,
        trigger_token: result.trigger_token ?? result.name,
        weight: 0.8,
      });
      void loadLoras();
    } catch { /* non-fatal */ } finally {
      setDownloading(null);
    }
  };

  const deleteLora = async (name: string) => {
    try {
      await axios.delete(`/api/lora/${name}`);
      void loadLoras();
    } catch { /* non-fatal */ }
  };

  return (
    <div className={`lora-manager ${embedded ? 'embedded' : ''}`} data-testid="lora-manager">
      {!embedded && <h2>LoRA Manager</h2>}

      {/* Filter tabs */}
      <div className="type-tabs" data-testid="lora-type-tabs">
        {['all', 'character', 'scene', 'style', 'identity'].map(t => (
          <button
            key={t}
            className={filterType === t ? 'active' : ''}
            onClick={() => setFilterType(t)}
            data-testid={`lora-filter-${t}`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Installed LoRAs */}
      <div className="lora-list" data-testid="lora-list">
        {loading && <p>Loading…</p>}
        {onSelectionChange && loras.length > 0 && (
          <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '8px' }}>
            {selectedNames.length === 0
              ? 'No LoRAs selected — none will be loaded during generation'
              : `${selectedNames.length} LoRA(s) selected for generation`}
          </div>
        )}
        {loras.map(lora => {
          const isSelected = selectedNames.includes(lora.name);
          return (
            <div
              key={lora.name}
              className={`lora-card status-${lora.status}`}
              data-testid={`lora-${lora.name}`}
              style={onSelectionChange ? { border: `1px solid ${isSelected ? '#e94560' : '#0f3460'}`, cursor: 'pointer' } : undefined}
              onClick={onSelectionChange ? () => toggleLora(lora.name) : undefined}
            >
              {onSelectionChange && (
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleLora(lora.name)}
                  onClick={e => e.stopPropagation()}
                  style={{ marginRight: '10px', accentColor: '#e94560' }}
                  data-testid={`select-lora-${lora.name}`}
                />
              )}
              <div className="lora-info" style={{ flex: 1 }}>
                <strong>{lora.name}</strong>
                <span className="lora-type">{lora.type}</span>
                <code className="trigger">{lora.trigger_token}</code>
                <span className={`status ${lora.status}`}>{lora.status}</span>
              </div>
              <button
                onClick={e => { e.stopPropagation(); void deleteLora(lora.name); }}
                className="delete-btn"
                data-testid={`delete-lora-${lora.name}`}
              >
                Remove
              </button>
            </div>
          );
        })}
        {loras.length === 0 && !loading && <p>No LoRAs installed.</p>}
      </div>

      {/* Online search */}
      <div className="online-search" data-testid="online-search">
        <h4>Find LoRAs Online</h4>
        <div className="search-row">
          <input
            type="text"
            placeholder="Search HuggingFace / Civitai…"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            data-testid="lora-search-input"
          />
          <button onClick={() => void searchOnline()} data-testid="lora-search-btn">Search</button>
        </div>
        {audioId && (
          <button onClick={() => void searchOnline()} data-testid="recommend-btn">
            Recommend for this project
          </button>
        )}
        <div className="search-results" data-testid="lora-search-results">
          {searchResults.map(r => (
            <div key={r.name} className="search-result" data-testid={`result-${r.name}`}>
              <strong>{r.name}</strong>
              <span>{r.source}</span>
              <span>{(r.confidence * 100).toFixed(0)}% match</span>
              <button
                onClick={() => void downloadLora(r)}
                disabled={downloading === r.name}
                data-testid={`download-${r.name}`}
              >
                {downloading === r.name ? 'Downloading…' : 'Download'}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
