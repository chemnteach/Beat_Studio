/**
 * SongAnalysis — displays the full analysis result for an uploaded song.
 * Shows BPM, key, mood, sections, energy, and lyrics.
 * Section editor allows full manual override: type, timestamps, lyrical content.
 */
import { useState } from 'react';
import type { SectionInfo, SongAnalysis as SongAnalysisType } from '../types';

interface Props {
  analysis: SongAnalysisType;
  onCreateVideo: () => void;
  onCreateMashup: () => void;
  onSectionsChange?: (sections: SectionInfo[]) => void;
}

const SECTION_TYPES = ['intro', 'verse', 'pre_chorus', 'chorus', 'bridge', 'outro'];

const SECTION_COLORS: Record<string, string> = {
  intro:      '#6c757d',
  verse:      '#0d6efd',
  pre_chorus: '#6610f2',
  chorus:     '#d63384',
  bridge:     '#fd7e14',
  outro:      '#6c757d',
};

function toSec(mmss: string): number {
  const parts = mmss.trim().split(':');
  if (parts.length === 2) {
    return (parseInt(parts[0], 10) || 0) * 60 + (parseFloat(parts[1]) || 0);
  }
  return parseFloat(mmss) || 0;
}

function toMMSS(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface EditRow {
  section_type: string;
  start: string;
  end: string;
  content: string;
}

function sectionsToRows(sections: SectionInfo[]): EditRow[] {
  return sections.map(s => ({
    section_type: s.section_type,
    start: toMMSS(s.start_sec),
    end:   toMMSS(s.end_sec),
    content: s.lyrical_content ?? '',
  }));
}

function rowsToSections(rows: EditRow[]): SectionInfo[] {
  return rows.map(r => {
    const start_sec = toSec(r.start);
    const end_sec   = toSec(r.end);
    return {
      section_type:     r.section_type,
      start_sec,
      end_sec,
      duration_sec:     Math.max(0, end_sec - start_sec),
      lyrical_content:  r.content,
      energy_level:     0,
      spectral_centroid: 0,
      tempo_stability:  0,
      vocal_density:    'medium',
      vocal_intensity:  0,
      emotional_tone:   'neutral',
      lyrical_function: 'narrative',
      themes:           [],
    };
  });
}

export function SongAnalysis({ analysis, onCreateVideo, onCreateMashup, onSectionsChange }: Props) {
  const [sections, setSections] = useState<SectionInfo[]>(() => analysis.sections ?? []);
  const [editMode, setEditMode] = useState(false);
  const [rows, setRows] = useState<EditRow[]>(() => sectionsToRows(analysis.sections ?? []));
  const [layoutName, setLayoutName] = useState('');

  const dur = analysis.duration_sec || 1;

  const applyEdits = () => {
    const updated = rowsToSections(rows);
    setSections(updated);
    setEditMode(false);
    onSectionsChange?.(updated);
  };

  const openEditor = () => {
    setRows(sectionsToRows(sections));
    setEditMode(true);
  };

  const updateRow = (i: number, field: keyof EditRow, value: string) => {
    setRows(prev => prev.map((r, idx) => idx === i ? { ...r, [field]: value } : r));
  };

  const addRow = () => {
    const last = rows[rows.length - 1];
    const newStart = last ? last.end : '0:00';
    setRows(prev => [...prev, { section_type: 'verse', start: newStart, end: newStart, content: '' }]);
  };

  const deleteRow = (i: number) => {
    setRows(prev => prev.filter((_, idx) => idx !== i));
  };

  const saveLayout = () => {
    const name = layoutName.trim() || analysis.title || 'sections';
    const payload = JSON.stringify({ name, song: analysis.title, sections: rows }, null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[^a-z0-9_-]/gi, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadLayout = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        if (data.name) setLayoutName(data.name);
        if (Array.isArray(data.sections)) {
          const loadedRows: EditRow[] = data.sections;
          const loadedSections = rowsToSections(loadedRows);
          // Update all three states atomically so openEditor can't clobber the loaded rows
          setRows(loadedRows);
          setSections(loadedSections);
          // Propagate to App immediately — user shouldn't need to click Apply after loading
          onSectionsChange?.(loadedSections);
          // Enter edit mode so the loaded sections are visible and can be verified
          setEditMode(true);
        }
      } catch { /* ignore bad files */ }
    };
    reader.readAsText(file);
    e.target.value = '';   // allow re-loading same file
  };

  return (
    <div className="song-analysis" data-testid="song-analysis">
      <h2>{analysis.title} — {analysis.artist}</h2>

      {/* Key metrics */}
      <div className="metrics" data-testid="analysis-metrics">
        <span className="metric"><strong>BPM</strong> {(analysis.bpm ?? 0).toFixed(1)}</span>
        <span className="metric"><strong>Key</strong> {analysis.key}</span>
        <span className="metric"><strong>Camelot</strong> {analysis.camelot}</span>
        <span className="metric"><strong>Duration</strong> {Math.round(analysis.duration_sec ?? 0)}s</span>
        <span className="metric"><strong>Energy</strong> {((analysis.energy_level ?? 0) * 100).toFixed(0)}%</span>
        {analysis.mood_summary && <span className="metric"><strong>Mood</strong> {analysis.mood_summary}</span>}
      </div>

      {/* Section timeline */}
      <div className="section-timeline" data-testid="section-timeline">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
          <h3 style={{ margin: 0 }}>Sections</h3>
          <button
            onClick={editMode ? applyEdits : openEditor}
            style={{ padding: '3px 10px', fontSize: '0.75rem', background: editMode ? '#198754' : '#0f3460' }}
            data-testid="edit-sections-btn"
          >
            {editMode ? '✓ Apply' : 'Edit Sections'}
          </button>
          {editMode && (
            <button
              onClick={() => setEditMode(false)}
              style={{ padding: '3px 10px', fontSize: '0.75rem', background: '#6c757d' }}
            >
              Cancel
            </button>
          )}
        </div>

        <div className="timeline-bar">
          {sections.map((sec, i) => {
            const widthPct = ((sec.end_sec - sec.start_sec) / dur) * 100;
            return (
              <div
                key={i}
                className="section-segment"
                style={{ width: `${widthPct}%`, backgroundColor: SECTION_COLORS[sec.section_type] ?? '#adb5bd' }}
                title={`${sec.section_type} (${toMMSS(sec.start_sec)}–${toMMSS(sec.end_sec)})`}
                data-testid={`section-${i}`}
              />
            );
          })}
        </div>
        <div className="section-legend">
          {sections.map((sec, i) => {
            const widthPct = ((sec.end_sec - sec.start_sec) / dur) * 100;
            return (
              <span
                key={i}
                className="section-label"
                style={{ width: `${widthPct}%`, color: SECTION_COLORS[sec.section_type] ?? '#adb5bd' }}
              >
                {widthPct > 4 ? sec.section_type : ''}
              </span>
            );
          })}
        </div>
      </div>

      {/* Section editor table */}
      {editMode && (
        <div style={{ marginTop: '16px' }} data-testid="section-editor">

          {/* Name + save/load bar */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap' }}>
            <input
              value={layoutName}
              onChange={e => setLayoutName(e.target.value)}
              placeholder="Layout name (e.g. Island Girl v2)"
              style={{ ...inputStyle, flex: '1', minWidth: '160px' }}
            />
            <button onClick={saveLayout} style={{ padding: '4px 12px', fontSize: '0.8rem', background: '#0f3460' }}>
              ↓ Save JSON
            </button>
            <label style={{ padding: '4px 12px', fontSize: '0.8rem', background: '#0f3460', borderRadius: '6px', cursor: 'pointer', color: '#e0e0e0' }}>
              ↑ Load JSON
              <input type="file" accept=".json" onChange={loadLayout} style={{ display: 'none' }} />
            </label>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ color: '#888', textTransform: 'uppercase', fontSize: '0.7rem' }}>
                  <th style={thStyle}>#</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Start</th>
                  <th style={thStyle}>End</th>
                  <th style={{ ...thStyle, minWidth: '260px' }}>Lyrics / Content</th>
                  <th style={thStyle}></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #0f3460', verticalAlign: 'top' }}>
                    <td style={{ ...tdStyle, paddingTop: '8px' }}>{i + 1}</td>
                    <td style={tdStyle}>
                      <select
                        value={row.section_type}
                        onChange={e => updateRow(i, 'section_type', e.target.value)}
                        style={inputStyle}
                      >
                        {SECTION_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                      </select>
                    </td>
                    <td style={tdStyle}>
                      <input
                        value={row.start}
                        onChange={e => updateRow(i, 'start', e.target.value)}
                        placeholder="0:00"
                        style={{ ...inputStyle, width: '60px' }}
                      />
                    </td>
                    <td style={tdStyle}>
                      <input
                        value={row.end}
                        onChange={e => updateRow(i, 'end', e.target.value)}
                        placeholder="0:00"
                        style={{ ...inputStyle, width: '60px' }}
                      />
                    </td>
                    <td style={tdStyle}>
                      <textarea
                        value={row.content}
                        onChange={e => updateRow(i, 'content', e.target.value)}
                        placeholder="Paste the full verse/chorus lyrics here…"
                        rows={3}
                        style={{ ...inputStyle, width: '100%', resize: 'vertical', fontFamily: 'inherit' }}
                      />
                    </td>
                    <td style={tdStyle}>
                      <button
                        onClick={() => deleteRow(i)}
                        style={{ padding: '2px 8px', background: '#6c757d', fontSize: '0.75rem' }}
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button
            onClick={addRow}
            style={{ marginTop: '8px', padding: '4px 12px', background: '#0f3460', fontSize: '0.8rem' }}
          >
            + Add Section
          </button>
        </div>
      )}

      {/* Lyrics preview */}
      {analysis.lyrics && (
        <div className="lyrics-preview" data-testid="lyrics-preview">
          <h3>Lyrics</h3>
          <pre>{analysis.lyrics.slice(0, 500)}{analysis.lyrics.length > 500 ? '…' : ''}</pre>
        </div>
      )}

      {/* Actions */}
      <div className="actions" data-testid="analysis-actions">
        <button onClick={onCreateVideo} data-testid="create-video-btn">Create Video</button>
        <button onClick={onCreateMashup} data-testid="create-mashup-btn">Create Mashup</button>
      </div>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: 'left', padding: '4px 8px', borderBottom: '1px solid #0f3460',
};
const tdStyle: React.CSSProperties = {
  padding: '4px 8px', verticalAlign: 'middle',
};
const inputStyle: React.CSSProperties = {
  background: '#16213e', color: '#e0e0e0', border: '1px solid #0f3460',
  borderRadius: '4px', padding: '3px 6px', fontSize: '0.8rem',
};
