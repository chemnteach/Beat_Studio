/**
 * StoryboardPreview — SDXL keyframe approval stage.
 *
 * Stage flow: lora → storyboard (here) → plan
 *
 * 1. User clicks "Generate Storyboard" → POST /api/video/storyboard/generate
 * 2. Component polls /api/tasks/{taskId} until complete
 * 3. Fetches /api/video/storyboard/{id}/images — shows carousel
 * 4. User can regenerate individual scenes (appends a new version)
 * 5. User selects preferred version per scene, clicks "Approve & Continue"
 * 6. POST /api/video/storyboard/{id}/approve → onApprove callback
 */
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import type { StoryboardImagesResponse, StoryboardSceneResult, StoryboardVersionEntry } from '../types';

interface StoryboardSceneInput {
  scene_idx: number;
  storyboard_prompt: string;
  positive_prompt: string;
}

interface Props {
  style: string;
  scenes: StoryboardSceneInput[];
  loraNames: string[];
  onApprove: (storyboardId: string, approvedPaths: Record<string, string>) => void;
  onBack: () => void;
}

type Phase = 'idle' | 'generating' | 'ready' | 'failed';

export function StoryboardPreview({ style, scenes, loraNames, onApprove, onBack }: Props) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [storyboardId, setStoryboardId] = useState<string | null>(null);
  const [imagesData, setImagesData] = useState<StoryboardImagesResponse | null>(null);
  // scene_idx → selected version number (1-indexed)
  const [selectedVersions, setSelectedVersions] = useState<Record<number, number>>({});
  const [activeScene, setActiveScene] = useState(0);
  // scene_idx → whether a regen is in progress
  const [regenRunning, setRegenRunning] = useState<Record<number, boolean>>({});
  const [error, setError] = useState('');
  const [isApproving, setIsApproving] = useState(false);

  const genPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const regenPollRefs = useRef<Record<number, ReturnType<typeof setInterval>>>({});

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (genPollRef.current) clearInterval(genPollRef.current);
      Object.values(regenPollRefs.current).forEach(clearInterval);
    };
  }, []);

  // Keyboard navigation: ← / → switch active scene when carousel is visible
  useEffect(() => {
    if (phase !== 'ready') return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft')  setActiveScene(a => Math.max(0, a - 1));
      if (e.key === 'ArrowRight') setActiveScene(a => Math.min((imagesData?.scenes.length ?? 1) - 1, a + 1));
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [phase, imagesData?.scenes.length]);

  // Fetch images and update selectedVersions.
  // If forceLatestFor is set, that scene's selection is always updated to the new latest.
  const fetchImages = async (id: string, forceLatestFor?: number) => {
    const { data } = await axios.get<StoryboardImagesResponse>(`/api/video/storyboard/${id}/images`);
    setImagesData(data);
    setSelectedVersions(prev => {
      const next = { ...prev };
      data.scenes.forEach(s => {
        const latest = s.versions[s.versions.length - 1]?.version;
        if (latest !== undefined && (next[s.scene_idx] === undefined || s.scene_idx === forceLatestFor)) {
          next[s.scene_idx] = latest;
        }
      });
      return next;
    });
  };

  const startGeneration = async () => {
    setPhase('generating');
    setError('');
    try {
      const { data } = await axios.post<{ task_id: string; storyboard_id: string }>(
        '/api/video/storyboard/generate',
        { style, lora_names: loraNames, scenes }
      );
      setStoryboardId(data.storyboard_id);

      genPollRef.current = setInterval(async () => {
        try {
          const { data: task } = await axios.get<{ status: string }>(`/api/tasks/${data.task_id}`);
          if (task.status === 'completed') {
            clearInterval(genPollRef.current!);
            genPollRef.current = null;
            await fetchImages(data.storyboard_id);
            setPhase('ready');
          } else if (task.status === 'failed' || task.status === 'cancelled') {
            clearInterval(genPollRef.current!);
            genPollRef.current = null;
            setPhase('failed');
            setError('Generation failed — check server logs and try again');
          }
        } catch { /* keep polling on transient errors */ }
      }, 2000);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? 'Failed to start storyboard generation';
      setError(msg);
      setPhase('failed');
    }
  };

  const regenerateScene = async (sceneIdx: number, promptOverride?: string) => {
    if (!storyboardId || regenRunning[sceneIdx]) return;
    setRegenRunning(prev => ({ ...prev, [sceneIdx]: true }));
    try {
      const { data } = await axios.post<{ task_id: string }>(
        `/api/video/storyboard/${storyboardId}/scene/${sceneIdx}/regenerate`,
        { positive_prompt: promptOverride ?? null, seed: null }
      );
      const taskId = data.task_id;
      regenPollRefs.current[sceneIdx] = setInterval(async () => {
        try {
          const { data: task } = await axios.get<{ status: string }>(`/api/tasks/${taskId}`);
          if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
            clearInterval(regenPollRefs.current[sceneIdx]);
            delete regenPollRefs.current[sceneIdx];
            if (task.status === 'completed') {
              await fetchImages(storyboardId!, sceneIdx);
            }
            setRegenRunning(prev => ({ ...prev, [sceneIdx]: false }));
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch {
      setRegenRunning(prev => ({ ...prev, [sceneIdx]: false }));
    }
  };

  const handleApprove = async () => {
    if (!storyboardId || !imagesData || isApproving) return;
    setIsApproving(true);
    setError('');
    try {
      const selections: Record<string, number> = {};
      imagesData.scenes.forEach(s => {
        const v = selectedVersions[s.scene_idx];
        if (v !== undefined) selections[String(s.scene_idx)] = v;
      });
      const { data } = await axios.post<{ storyboard_id: string; approved_paths: Record<string, string> }>(
        `/api/video/storyboard/${storyboardId}/approve`,
        { selections }
      );
      onApprove(data.storyboard_id, data.approved_paths);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? 'Approval failed';
      setError(msg);
    } finally {
      setIsApproving(false);
    }
  };

  // ── Render: idle (pre-generate) ────────────────────────────────────────────

  if (phase === 'idle') {
    return (
      <div data-testid="stage-storyboard-idle">
        <h3 style={{ marginBottom: '8px' }}>Storyboard Preview</h3>
        <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '20px' }}>
          Generate SDXL keyframe images for all {scenes.length} scenes using the <strong style={{ color: '#e0e0e0' }}>{style}</strong> style
          {loraNames.length > 0 && ` + ${loraNames.join(', ')}`}.
          Review, optionally regenerate individual scenes, then approve to continue.
        </p>
        {error && <p className="error">{error}</p>}
        <div className="stage-nav">
          <button onClick={onBack}>← Back to LoRA</button>
          <button onClick={() => void startGeneration()} data-testid="generate-storyboard-btn">
            Generate Storyboard →
          </button>
        </div>
      </div>
    );
  }

  // ── Render: generating ─────────────────────────────────────────────────────

  if (phase === 'generating') {
    return (
      <div data-testid="stage-storyboard-generating" style={{ textAlign: 'center', padding: '40px 0' }}>
        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>🎨</div>
        <p style={{ fontSize: '1rem', marginBottom: '8px' }}>
          Generating {scenes.length} storyboard frames with SDXL…
        </p>
        <p style={{ color: '#888', fontSize: '0.82rem' }}>
          Takes ~30–90 seconds depending on GPU and number of scenes
        </p>
      </div>
    );
  }

  // ── Render: failed ─────────────────────────────────────────────────────────

  if (phase === 'failed') {
    return (
      <div data-testid="stage-storyboard-failed">
        <p className="error">{error || 'Storyboard generation failed'}</p>
        <div className="stage-nav">
          <button onClick={onBack}>← Back to LoRA</button>
          <button onClick={() => { setPhase('idle'); setError(''); }}>Try Again</button>
        </div>
      </div>
    );
  }

  // ── Render: ready (carousel) ───────────────────────────────────────────────

  const activeSceneData: StoryboardSceneResult | undefined =
    imagesData?.scenes.find(s => s.scene_idx === activeScene);

  const selectedVersion = activeSceneData
    ? (selectedVersions[activeScene] ?? activeSceneData.versions[activeSceneData.versions.length - 1]?.version ?? 1)
    : 1;

  const selectedEntry: StoryboardVersionEntry | undefined =
    activeSceneData?.versions.find(v => v.version === selectedVersion);

  const isActiveRegen = regenRunning[activeScene] ?? false;

  return (
    <div data-testid="stage-storyboard-ready">
      <h3 style={{ marginBottom: '12px' }}>Storyboard Preview</h3>

      {/* ── Main view: large image + controls ── */}
      <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', alignItems: 'flex-start' }}>

        {/* Large image */}
        <div style={{ flex: 1, position: 'relative' }}>
          {isActiveRegen && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
              justifyContent: 'center', background: 'rgba(22,33,62,0.75)', borderRadius: '8px', zIndex: 1,
            }}>
              <span style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>⟳ Generating…</span>
            </div>
          )}
          {selectedEntry ? (
            <img
              src={selectedEntry.url}
              alt={`Scene ${activeScene + 1} v${selectedVersion}`}
              style={{
                width: '100%', borderRadius: '8px',
                border: `2px solid ${isActiveRegen ? '#555' : '#0f3460'}`,
                display: 'block',
              }}
            />
          ) : (
            <div style={{
              background: '#16213e', borderRadius: '8px', height: '220px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#555',
            }}>
              No image
            </div>
          )}
        </div>

        {/* Right panel: scene info + version picker + regen */}
        <div style={{ width: '190px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Scene {activeScene + 1} of {imagesData?.scenes.length}
          </div>

          {activeSceneData && (
            <div style={{ fontSize: '0.78rem', color: '#aaa', lineHeight: 1.5 }}>
              {activeSceneData.storyboard_prompt.length > 120
                ? activeSceneData.storyboard_prompt.substring(0, 117) + '…'
                : activeSceneData.storyboard_prompt}
            </div>
          )}

          {/* Version picker — shown when scene has multiple versions or a regen is pending */}
          {activeSceneData && (activeSceneData.versions.length > 1 || isActiveRegen) && (
            <div>
              <div style={{ fontSize: '0.7rem', color: '#666', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '6px' }}>
                Versions
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {activeSceneData.versions.map(v => (
                  <button
                    key={v.version}
                    onClick={() => setSelectedVersions(prev => ({ ...prev, [activeScene]: v.version }))}
                    title={`Seed: ${v.seed}`}
                    style={{
                      padding: '3px 9px',
                      fontSize: '0.75rem',
                      background: selectedVersion === v.version ? '#e94560' : '#16213e',
                      border: `1px solid ${selectedVersion === v.version ? '#e94560' : '#0f3460'}`,
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    v{v.version}
                  </button>
                ))}
                {/* Optimistic placeholder — visible while a new version is being generated */}
                {isActiveRegen && (
                  <button
                    disabled
                    style={{
                      padding: '3px 9px',
                      fontSize: '0.75rem',
                      background: '#16213e',
                      border: '1px dashed #555',
                      borderRadius: '4px',
                      color: '#888',
                      cursor: 'default',
                    }}
                  >
                    v{(activeSceneData.versions[activeSceneData.versions.length - 1]?.version ?? 0) + 1} ⟳
                  </button>
                )}
              </div>
            </div>
          )}

          <button
            onClick={() => void regenerateScene(activeScene)}
            disabled={isActiveRegen}
            style={{ fontSize: '0.8rem', background: '#0f3460' }}
          >
            {isActiveRegen ? '⟳ Generating…' : '↺ Regenerate Scene'}
          </button>

          {/* Scene navigation arrows */}
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={() => setActiveScene(a => Math.max(0, a - 1))}
              disabled={activeScene === 0}
              style={{ flex: 1, fontSize: '0.8rem', background: '#0f3460' }}
            >
              ← Prev
            </button>
            <button
              onClick={() => setActiveScene(a => Math.min((imagesData?.scenes.length ?? 1) - 1, a + 1))}
              disabled={activeScene === (imagesData?.scenes.length ?? 1) - 1}
              style={{ flex: 1, fontSize: '0.8rem', background: '#0f3460' }}
            >
              Next →
            </button>
          </div>
        </div>
      </div>

      {/* ── Thumbnail strip ── */}
      <div style={{
        display: 'flex',
        gap: '6px',
        overflowX: 'auto',
        padding: '10px 0',
        marginBottom: '20px',
        borderTop: '1px solid #0f3460',
        borderBottom: '1px solid #0f3460',
      }}>
        {imagesData?.scenes.map(s => {
          const selV = selectedVersions[s.scene_idx] ?? s.versions[s.versions.length - 1]?.version ?? 1;
          const vEntry = s.versions.find(v => v.version === selV) ?? s.versions[s.versions.length - 1];
          const isActive = s.scene_idx === activeScene;
          const isRegen = regenRunning[s.scene_idx] ?? false;
          return (
            <div
              key={s.scene_idx}
              onClick={() => setActiveScene(s.scene_idx)}
              title={`Scene ${s.scene_idx + 1}`}
              style={{
                flexShrink: 0,
                width: '88px',
                cursor: 'pointer',
                border: `2px solid ${isActive ? '#e94560' : '#0f3460'}`,
                borderRadius: '6px',
                overflow: 'hidden',
                position: 'relative',
                opacity: isRegen ? 0.55 : 1,
                transition: 'opacity 0.2s',
              }}
            >
              {vEntry ? (
                <img src={vEntry.url} alt={`Scene ${s.scene_idx + 1}`} style={{ width: '100%', display: 'block' }} />
              ) : (
                <div style={{ height: '56px', background: '#16213e' }} />
              )}
              <div style={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                background: 'rgba(0,0,0,0.65)',
                fontSize: '0.62rem', padding: '2px 4px',
                color: isActive ? '#e94560' : '#e0e0e0',
                textAlign: 'center',
                fontWeight: isActive ? 'bold' : 'normal',
              }}>
                {isRegen ? '⟳' : `#${s.scene_idx + 1}`}
              </div>
            </div>
          );
        })}
      </div>

      {error && <p className="error" style={{ marginBottom: '12px' }}>{error}</p>}

      <div className="stage-nav">
        <button onClick={onBack}>← Back to LoRA</button>
        <button
          onClick={() => void handleApprove()}
          disabled={isApproving}
          data-testid="approve-storyboard-btn"
          style={{ background: '#2ecc71', color: '#0a1a0a' }}
        >
          {isApproving ? 'Approving…' : 'Approve & Continue →'}
        </button>
      </div>
    </div>
  );
}
