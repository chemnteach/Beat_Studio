/**
 * MashupWorkshop — song library browser, compatibility matcher, and mashup creation.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { SongEntry, MatchResult, MashupType } from '../types';

interface Props {
  initialSongId?: string;
  onMashupCreated: (taskId: string) => void;
  onMakeVideo: (audioId: string) => void;
  onMakeDJVideo: (audioId: string) => void;
}

const MASHUP_TYPES: { value: MashupType; label: string }[] = [
  { value: 'classic', label: 'Classic (vocal A + instrumental B)' },
  { value: 'stem_swap', label: 'Stem Swap (mix from 3+ songs)' },
  { value: 'energy_match', label: 'Energy Match' },
  { value: 'adaptive_harmony', label: 'Adaptive Harmony' },
  { value: 'theme_fusion', label: 'Theme Fusion' },
  { value: 'semantic_aligned', label: 'Semantic Aligned' },
  { value: 'role_aware', label: 'Role Aware' },
  { value: 'conversational', label: 'Conversational' },
];

export function MashupWorkshop({ initialSongId, onMashupCreated, onMakeVideo, onMakeDJVideo }: Props) {
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [selectedA, setSelectedA] = useState<string>(initialSongId ?? '');
  const [selectedB, setSelectedB] = useState<string>('');
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [mashupType, setMashupType] = useState<MashupType>('classic');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    void fetchLibrary();
  }, []);

  const fetchLibrary = async () => {
    try {
      const { data } = await axios.get<{ songs: SongEntry[] }>('/api/mashup/library');
      setSongs(data.songs);
    } catch { /* non-fatal */ }
  };

  const findMatches = async (songId: string) => {
    setLoading(true);
    try {
      const { data } = await axios.post<{ matches: MatchResult[] }>('/api/mashup/match', null, {
        params: { song_id: songId, criteria: 'hybrid', top: 5 },
      });
      setMatches(data.matches);
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  const handleSongASelect = (id: string) => {
    setSelectedA(id);
    if (id) void findMatches(id);
  };

  const searchLibrary = async () => {
    if (!searchQuery.trim()) return;
    try {
      const { data } = await axios.get<{ results: SongEntry[] }>('/api/mashup/library/search', {
        params: { q: searchQuery },
      });
      setSongs(data.results);
    } catch { /* non-fatal */ }
  };

  const createMashup = async () => {
    if (!selectedA || !selectedB) return;
    setCreating(true);
    try {
      const { data } = await axios.post<{ task_id: string }>('/api/mashup/create', {
        song_a_id: selectedA,
        song_b_id: selectedB,
        mashup_type: mashupType,
      });
      onMashupCreated(data.task_id);
    } catch { /* non-fatal */ } finally {
      setCreating(false);
    }
  };

  return (
    <div className="mashup-workshop" data-testid="mashup-workshop">
      <h2>Mashup Workshop</h2>

      {/* Library search */}
      <div className="library-search" data-testid="library-search">
        <input
          type="text"
          placeholder="Search library…"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && void searchLibrary()}
          data-testid="library-search-input"
        />
        <button onClick={() => void searchLibrary()}>Search</button>
      </div>

      {/* Song A selector */}
      <div className="song-selector" data-testid="song-a-selector">
        <label>Song A (source vocal)</label>
        <select value={selectedA} onChange={e => handleSongASelect(e.target.value)} data-testid="song-a-select">
          <option value="">Select a song…</option>
          {songs.map(s => (
            <option key={s.id} value={s.id}>{s.artist} — {s.title}</option>
          ))}
        </select>
      </div>

      {/* Matches / Song B */}
      {matches.length > 0 && (
        <div className="matches" data-testid="match-results">
          <h3>Compatible Matches</h3>
          {matches.map(m => (
            <div
              key={m.song.id}
              className={`match-card ${selectedB === m.song.id ? 'selected' : ''}`}
              onClick={() => setSelectedB(m.song.id)}
              data-testid={`match-${m.song.id}`}
            >
              <strong>{m.song.artist} — {m.song.title}</strong>
              <span>Score: {(m.score * 100).toFixed(0)}%</span>
              <span>Recommended: {m.recommended_type}</span>
            </div>
          ))}
        </div>
      )}

      {loading && <p>Finding matches…</p>}

      {/* Mashup type */}
      <div className="mashup-type-selector" data-testid="mashup-type-selector">
        <label>Mashup Type</label>
        <select value={mashupType} onChange={e => setMashupType(e.target.value as MashupType)} data-testid="mashup-type-select">
          {MASHUP_TYPES.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
      </div>

      <div className="actions">
        <button
          onClick={() => void createMashup()}
          disabled={!selectedA || !selectedB || creating}
          data-testid="create-mashup-btn"
        >
          {creating ? 'Creating…' : 'Create Mashup'}
        </button>
        {selectedA && (
          <>
            <button onClick={() => onMakeVideo(selectedA)} data-testid="make-video-btn">Make Video</button>
            <button onClick={() => onMakeDJVideo(selectedA)} data-testid="make-dj-video-btn">Make DJ Video</button>
          </>
        )}
      </div>
    </div>
  );
}
