/**
 * AudioUpload — drag-and-drop audio file upload + YouTube URL input.
 * Supported formats: mp3, wav, flac, m4a.
 */
import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import axios from 'axios';

interface Props {
  onUploadComplete: (audioId: string, filename: string) => void;
}

const ACCEPTED = ['.mp3', '.wav', '.flac', '.m4a'];

export function AudioUpload({ onUploadComplete }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadFile = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const { data } = await axios.post<{ audio_id: string; filename: string }>(
        '/api/audio/upload',
        form,
      );
      onUploadComplete(data.audio_id, data.filename);
    } catch {
      setError('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) void uploadFile(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void uploadFile(file);
  };

  const handleYouTubeIngest = async () => {
    if (!youtubeUrl.trim()) return;
    setUploading(true);
    setError(null);
    try {
      const { data } = await axios.post<{ task_id: string }>('/api/mashup/ingest', {
        source: youtubeUrl,
      });
      onUploadComplete(data.task_id, youtubeUrl);
    } catch {
      setError('YouTube ingest failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="audio-upload" data-testid="audio-upload">
      {/* Drag-drop zone */}
      <div
        className={`drop-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        aria-label="Drop audio file here or click to browse"
      >
        {uploading
          ? 'Uploading...'
          : dragging
          ? 'Drop it!'
          : `Drop audio here or click to browse (${ACCEPTED.join(', ')})`}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        data-testid="file-input"
      />

      {/* YouTube URL */}
      <div className="youtube-input">
        <input
          type="url"
          placeholder="Or paste a YouTube URL…"
          value={youtubeUrl}
          onChange={e => setYoutubeUrl(e.target.value)}
          data-testid="youtube-url-input"
        />
        <button onClick={() => void handleYouTubeIngest()} disabled={!youtubeUrl || uploading}>
          Ingest
        </button>
      </div>

      {error && <p className="error" data-testid="upload-error">{error}</p>}
    </div>
  );
}
