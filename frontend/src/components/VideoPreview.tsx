/**
 * VideoPreview — final video player with download options and scene frame inspector.
 */
import { useState } from 'react';
import { ClipFrameViewer } from './ClipFrameViewer';

interface Props {
  videoId: string;
  videoUrl?: string;
  platforms?: string[];
}

const DEFAULT_PLATFORMS = ['youtube', 'tiktok', 'reels', 'shorts'];

export function VideoPreview({ videoId, videoUrl, platforms = DEFAULT_PLATFORMS }: Props) {
  const baseDownloadUrl = `/api/video/download/${videoId}`;
  const [showFrames, setShowFrames] = useState(false);

  return (
    <div className="video-preview" data-testid="video-preview">
      <h3>Your Video is Ready!</h3>

      {videoUrl ? (
        <video
          src={videoUrl}
          controls
          className="video-player"
          data-testid="video-player"
        />
      ) : (
        <div className="video-placeholder" data-testid="video-placeholder">
          Video preview not available in browser. Download to view.
        </div>
      )}

      <div className="download-options" data-testid="download-options">
        <h4>Download</h4>
        {platforms.map(platform => (
          <a
            key={platform}
            href={`${baseDownloadUrl}?platform=${platform}`}
            download
            className="download-btn"
            data-testid={`download-${platform}`}
          >
            {platform.charAt(0).toUpperCase() + platform.slice(1)}
          </a>
        ))}
      </div>

      <div style={{ marginTop: '20px', borderTop: '1px solid #1a1a2e', paddingTop: '16px' }}>
        <button
          onClick={() => setShowFrames(v => !v)}
          style={{ background: '#0f3460', fontSize: '0.85rem' }}
          data-testid="inspect-frames-btn"
        >
          {showFrames ? '▲ Hide Scene Frames' : '▼ Inspect Scene Frames'}
        </button>
        {showFrames && (
          <div style={{ marginTop: '12px' }}>
            <ClipFrameViewer videoId={videoId} />
          </div>
        )}
      </div>
    </div>
  );
}
