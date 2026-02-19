/**
 * NovaFadeStudio — Nova Fade DJ video workflow.
 * Manages canonical image generation, identity/style LoRA status,
 * drift testing, and DJ video generation.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { NovaFadeStatus, NovaFadeTheme } from '../types';

interface Props {
  mashupId?: string;
}

const THEMES: { value: NovaFadeTheme; label: string; description: string }[] = [
  { value: 'sponsor_neon', label: 'Sponsor Neon', description: 'Neon-lit club with brand sponsor lighting' },
  { value: 'award_elegant', label: 'Award Elegant', description: 'Clean award-show studio, premium feel' },
  { value: 'mashup_chaos', label: 'Mashup Chaos', description: 'High-energy chaos, multiple visual layers' },
  { value: 'chill_lofi', label: 'Chill Lofi', description: 'Lo-fi aesthetic, warm tones, intimate' },
];

export function NovaFadeStudio({ mashupId }: Props) {
  const [status, setStatus] = useState<NovaFadeStatus | null>(null);
  const [selectedTheme, setSelectedTheme] = useState<NovaFadeTheme>('sponsor_neon');
  const [generating, setGenerating] = useState(false);
  const [trainingIdentity, setTrainingIdentity] = useState(false);
  const [runningDrift, setRunningDrift] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);

  useEffect(() => {
    void fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const { data } = await axios.get<NovaFadeStatus>('/api/nova-fade/status');
      setStatus(data);
    } catch { /* non-fatal */ }
  };

  const generateCanonical = async () => {
    try {
      await axios.post('/api/nova-fade/generate-canonical', { num_images: 20 });
      void fetchStatus();
    } catch { /* non-fatal */ }
  };

  const trainIdentityLoRA = async () => {
    setTrainingIdentity(true);
    try {
      const { data } = await axios.post<{ task_id: string }>('/api/nova-fade/train-identity-lora');
      setTaskId(data.task_id);
    } catch { /* non-fatal */ } finally {
      setTrainingIdentity(false);
    }
  };

  const runDriftTest = async () => {
    setRunningDrift(true);
    try {
      const { data } = await axios.post<{ task_id: string }>('/api/nova-fade/drift-test', {
        lora_path: 'output/loras/novafade_id_v1.safetensors',
      });
      setTaskId(data.task_id);
    } catch { /* non-fatal */ } finally {
      setRunningDrift(false);
    }
  };

  const generateDJVideo = async () => {
    if (!mashupId) return;
    setGenerating(true);
    try {
      const { data } = await axios.post<{ task_id: string }>('/api/nova-fade/dj-video', {
        mashup_id: mashupId,
        theme: selectedTheme,
      });
      setTaskId(data.task_id);
    } catch { /* non-fatal */ } finally {
      setGenerating(false);
    }
  };

  const StatusBadge = ({ ok }: { ok: boolean }) => (
    <span className={`status-badge ${ok ? 'ready' : 'missing'}`} data-testid="status-badge">
      {ok ? 'Ready' : 'Not Ready'}
    </span>
  );

  return (
    <div className="nova-fade-studio" data-testid="nova-fade-studio">
      <h2>Nova Fade DJ Studio</h2>
      <p className="constitution-note">
        Constitution v{status?.constitution_version ?? '1.0'} — all prompts enforced automatically.
      </p>

      {/* Status panel */}
      {status && (
        <div className="nova-status-panel" data-testid="nova-status-panel">
          <div className="status-row">
            <span>Identity LoRA</span>
            <StatusBadge ok={status.identity_lora === 'available'} />
          </div>
          <div className="status-row">
            <span>Style LoRA</span>
            <StatusBadge ok={status.style_lora === 'available'} />
          </div>
          <div className="status-row">
            <span>Canonical Images</span>
            <span data-testid="canonical-count">{status.canonical_images}</span>
          </div>
          <div className="status-row">
            <span>Last Drift Test</span>
            <span data-testid="last-drift">{status.last_drift_test ?? 'Never'}</span>
          </div>
        </div>
      )}

      {/* Setup actions */}
      <div className="setup-actions" data-testid="setup-actions">
        <h3>Setup</h3>
        <button
          onClick={() => void generateCanonical()}
          data-testid="generate-canonical-btn"
          disabled={status?.canonical_images ? status.canonical_images >= 15 : false}
        >
          Generate Canonical Images (SDXL)
        </button>
        <button
          onClick={() => void trainIdentityLoRA()}
          disabled={trainingIdentity || !status?.canonical_images}
          data-testid="train-identity-btn"
        >
          {trainingIdentity ? 'Training…' : 'Train Identity LoRA'}
        </button>
        <button
          onClick={() => void runDriftTest()}
          disabled={runningDrift || status?.identity_lora !== 'available'}
          data-testid="run-drift-test-btn"
        >
          {runningDrift ? 'Running…' : 'Run Drift Test'}
        </button>
      </div>

      {/* DJ video generation */}
      <div className="dj-video-section" data-testid="dj-video-section">
        <h3>Generate DJ Video</h3>
        {!mashupId && (
          <p className="no-mashup-warning">Select a mashup in the Mashup Workshop first.</p>
        )}
        <div className="theme-grid" data-testid="theme-grid">
          {THEMES.map(t => (
            <div
              key={t.value}
              className={`theme-card ${selectedTheme === t.value ? 'selected' : ''}`}
              onClick={() => setSelectedTheme(t.value)}
              data-testid={`theme-${t.value}`}
            >
              <h4>{t.label}</h4>
              <p>{t.description}</p>
            </div>
          ))}
        </div>
        <button
          onClick={() => void generateDJVideo()}
          disabled={!mashupId || generating || status?.identity_lora !== 'available'}
          data-testid="generate-dj-video-btn"
        >
          {generating ? 'Generating…' : 'Generate DJ Video'}
        </button>
      </div>

      {taskId && (
        <div className="task-status" data-testid="task-status">
          Task started: {taskId}
        </div>
      )}
    </div>
  );
}
