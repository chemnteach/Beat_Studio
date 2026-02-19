/**
 * VideoPreview — final video player with download options.
 */
interface Props {
  videoId: string;
  videoUrl?: string;
  platforms?: string[];
}

const DEFAULT_PLATFORMS = ['youtube', 'tiktok', 'reels', 'shorts'];

export function VideoPreview({ videoId, videoUrl, platforms = DEFAULT_PLATFORMS }: Props) {
  const baseDownloadUrl = `/api/video/download/${videoId}`;

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
    </div>
  );
}
