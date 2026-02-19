/**
 * SongAnalysis — displays the full analysis result for an uploaded song.
 * Shows BPM, key, mood, sections, energy, and lyrics.
 */
import type { SongAnalysis as SongAnalysisType } from '../types';

interface Props {
  analysis: SongAnalysisType;
  onCreateVideo: () => void;
  onCreateMashup: () => void;
}

export function SongAnalysis({ analysis, onCreateVideo, onCreateMashup }: Props) {
  const sectionColors: Record<string, string> = {
    intro: '#6c757d',
    verse: '#0d6efd',
    pre_chorus: '#6610f2',
    chorus: '#d63384',
    bridge: '#fd7e14',
    outro: '#6c757d',
  };

  return (
    <div className="song-analysis" data-testid="song-analysis">
      <h2>{analysis.title} — {analysis.artist}</h2>

      {/* Key metrics */}
      <div className="metrics" data-testid="analysis-metrics">
        <span className="metric"><strong>BPM</strong> {analysis.bpm.toFixed(1)}</span>
        <span className="metric"><strong>Key</strong> {analysis.key}</span>
        <span className="metric"><strong>Camelot</strong> {analysis.camelot}</span>
        <span className="metric"><strong>Duration</strong> {Math.round(analysis.duration)}s</span>
        <span className="metric"><strong>Energy</strong> {(analysis.energy * 100).toFixed(0)}%</span>
        <span className="metric"><strong>Mood</strong> {analysis.mood_summary}</span>
      </div>

      {/* Section timeline */}
      <div className="section-timeline" data-testid="section-timeline">
        <h3>Sections</h3>
        <div className="timeline-bar">
          {analysis.sections.map((sec, i) => {
            const widthPct = ((sec.end - sec.start) / analysis.duration) * 100;
            return (
              <div
                key={i}
                className="section-segment"
                style={{
                  width: `${widthPct}%`,
                  backgroundColor: sectionColors[sec.section_type] ?? '#adb5bd',
                }}
                title={`${sec.section_type} (${sec.start.toFixed(1)}s–${sec.end.toFixed(1)}s)`}
                data-testid={`section-${i}`}
              />
            );
          })}
        </div>
        <div className="section-legend">
          {analysis.sections.map((sec, i) => (
            <span key={i} className="section-label" style={{ color: sectionColors[sec.section_type] }}>
              {sec.section_type}
            </span>
          ))}
        </div>
      </div>

      {/* Lyrics preview */}
      {analysis.lyrics && (
        <div className="lyrics-preview" data-testid="lyrics-preview">
          <h3>Lyrics</h3>
          <pre>{analysis.lyrics.slice(0, 500)}{analysis.lyrics.length > 500 ? '…' : ''}</pre>
        </div>
      )}

      {/* Actions */}
      <div className="actions" data-testid="analysis-actions">
        <button onClick={onCreateVideo} data-testid="create-video-btn">
          Create Video
        </button>
        <button onClick={onCreateMashup} data-testid="create-mashup-btn">
          Create Mashup
        </button>
      </div>
    </div>
  );
}
