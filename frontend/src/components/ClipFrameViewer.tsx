/**
 * ClipFrameViewer — displays extracted first frames from all generated clips.
 * Used for diagnosing style quality, LoRA activation, and generation fidelity.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';

interface Props {
  videoId?: string;  // If provided, shows only frames from that run
  onClose?: () => void;
}

export function ClipFrameViewer({ videoId, onClose }: Props) {
  const [frames, setFrames] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const url = videoId ? `/api/video/frames?video_id=${videoId}` : '/api/video/frames';
    axios.get<{ frames: string[] }>(url)
      .then(({ data }) => { setFrames(data.frames); setLoading(false); })
      .catch(() => { setError('Could not load frames'); setLoading(false); });
  }, [videoId]);

  const clipId = (filename: string) => filename.replace('_frame0.png', '').replace('clip_', '');

  return (
    <div style={{ padding: '16px' }} data-testid="clip-frame-viewer">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ margin: 0 }}>Scene Frames <span style={{ fontSize: '0.75rem', color: '#888', fontWeight: 'normal' }}>({frames.length} clips)</span></h3>
        {onClose && <button onClick={onClose} style={{ background: '#333', padding: '4px 12px', fontSize: '0.8rem' }}>✕ Close</button>}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading frames…</p>}
      {error && <p style={{ color: '#e94560' }}>{error}</p>}

      {/* Thumbnail grid */}
      {!loading && frames.length === 0 && (
        <p style={{ color: '#888' }}>No frames found. Run generation first.</p>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '10px' }}>
        {frames.map(f => (
          <div
            key={f}
            onClick={() => setSelected(f)}
            style={{
              cursor: 'pointer',
              border: `2px solid ${selected === f ? '#e94560' : '#0f3460'}`,
              borderRadius: '6px',
              overflow: 'hidden',
              background: '#0a0a1a',
            }}
          >
            <img
              src={`/api/video/frames/${f}`}
              alt={f}
              style={{ width: '100%', display: 'block' }}
              loading="lazy"
            />
            <div style={{ padding: '4px 6px', fontSize: '0.7rem', color: '#888', fontFamily: 'monospace' }}>
              clip_{clipId(f)}
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {selected && (
        <div
          onClick={() => setSelected(null)}
          style={{
            position: 'fixed', inset: 0,
            background: 'rgba(0,0,0,0.88)',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            zIndex: 1000, cursor: 'zoom-out',
          }}
        >
          <img
            src={`/api/video/frames/${selected}`}
            alt={selected}
            style={{ maxWidth: '90vw', maxHeight: '85vh', borderRadius: '8px', boxShadow: '0 0 40px #000' }}
          />
          <div style={{ marginTop: '12px', fontSize: '0.8rem', color: '#aaa', fontFamily: 'monospace' }}>
            {selected} — click anywhere to close
          </div>
        </div>
      )}
    </div>
  );
}
