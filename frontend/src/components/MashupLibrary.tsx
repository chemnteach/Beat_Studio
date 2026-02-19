/**
 * MashupLibrary — browse and search the ChromaDB song library.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { SongEntry } from '../types';

interface Props {
  onSongSelect: (song: SongEntry) => void;
}

export function MashupLibrary({ onSongSelect }: Props) {
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [query, setQuery] = useState('');
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void loadAll();
  }, []);

  const loadAll = async () => {
    setLoading(true);
    try {
      const { data } = await axios.get<{ songs: SongEntry[]; total: number }>('/api/mashup/library');
      setSongs(data.songs);
      setTotal(data.total);
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  const search = async () => {
    if (!query.trim()) { void loadAll(); return; }
    setLoading(true);
    try {
      const { data } = await axios.get<{ results: SongEntry[] }>('/api/mashup/library/search', {
        params: { q: query },
      });
      setSongs(data.results);
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mashup-library" data-testid="mashup-library">
      <h3>Song Library ({total} songs)</h3>
      <div className="search-bar">
        <input
          type="text"
          placeholder="Search by mood, genre, or title…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && void search()}
          data-testid="library-query-input"
        />
        <button onClick={() => void search()}>Search</button>
        <button onClick={() => { setQuery(''); void loadAll(); }}>Clear</button>
      </div>
      {loading && <p>Loading…</p>}
      <div className="song-list" data-testid="song-list">
        {songs.map(song => (
          <div
            key={song.id}
            className="song-card"
            onClick={() => onSongSelect(song)}
            data-testid={`song-${song.id}`}
          >
            <strong>{song.artist} — {song.title}</strong>
            <span>{song.bpm.toFixed(0)} BPM · {song.key} · {Math.round(song.duration)}s</span>
            <em>{song.mood_summary}</em>
          </div>
        ))}
        {songs.length === 0 && !loading && <p>No songs in library. Upload some tracks!</p>}
      </div>
    </div>
  );
}
