/**
 * StoryboardPreview — SDXL keyframe approval stage.
 *
 * Stage flow: lora → storyboard (here) → plan
 *
 * 1. User clicks "Generate Storyboard" → POST /api/video/storyboard/generate
 * 2. Component polls /api/tasks/{taskId} until complete (+ secondary image poll for progress)
 * 3. Fetches /api/video/storyboard/{id}/images — shows carousel
 * 4. Per-scene LoRA weight sliders let the user dial back LoRA influence before regen
 * 5. User can regenerate individual scenes (appends a new version)
 * 6. User selects preferred version per scene, clicks "Approve & Continue"
 * 7. POST /api/video/storyboard/{id}/approve → onApprove callback
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import type { StoryboardImagesResponse, StoryboardSceneResult, StoryboardVersionEntry } from '../types';

interface StoryboardSceneInput {
  scene_idx: number;
  storyboard_prompt: string;
  positive_prompt: string;
}

interface LoraDetail {
  name: string;
  trigger_token: string;
  type: string;   // "character" | "style" | "scene" | "identity"
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
  // LoRA details fetched from /api/lora/list (filtered to loraNames prop)
  const [loraDetails, setLoraDetails] = useState<LoraDetail[]>([]);
  // scene_idx → { lora_name: weight }; default 0.7 per LoRA
  const [sceneLoraWeights, setSceneLoraWeights] = useState<Record<number, Record<string, number>>>({});
  // scene_idx → user-edited positive_prompt; undefined = keep original
  const [promptEdits, setPromptEdits] = useState<Record<number, string>>({});
  // how many scenes have completed images (for generating phase progress bar)
  const [progressSceneCount, setProgressSceneCount] = useState(0);

  const genPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const imgProgressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const regenPollRefs = useRef<Record<number, ReturnType<typeof setInterval>>>({});

  // Fetch LoRA details once on mount (non-fatal)
  useEffect(() => {
    if (loraNames.length === 0) return;
    axios.get<{ loras: LoraDetail[] }>('/api/lora/list')
      .then(({ data }) => {
        const filtered = data.loras.filter(l => loraNames.includes(l.name));
        setLoraDetails(filtered);
      })
      .catch(() => { /* non-fatal */ });
  }, [loraNames]);

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (genPollRef.current) clearInterval(genPollRef.current);
      if (imgProgressPollRef.current) clearInterval(imgProgressPollRef.current);
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

  // Return LoRA details active for a given scene (trigger token appears in positive_prompt)
  const getActiveLorasForScene = useCallback(
    (scene: StoryboardSceneResult | StoryboardSceneInput): LoraDetail[] => {
      const prompt = scene.positive_prompt.toLowerCase();
      return loraDetails.filter(l =>
        l.trigger_token ? prompt.includes(l.trigger_token.toLowerCase()) : true
      );
    },
    [loraDetails],
  );

  // Get or initialise the weight map for a scene, defaulting each active LoRA to 0.7
  const getSceneWeights = useCallback(
    (sceneIdx: number, activeLoras: LoraDetail[]): Record<string, number> => {
      const existing = sceneLoraWeights[sceneIdx] ?? {};
      const result: Record<string, number> = {};
      activeLoras.forEach(l => {
        result[l.name] = existing[l.name] ?? 0.7;
      });
      return result;
    },
    [sceneLoraWeights],
  );

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
    setProgressSceneCount(0);
    setError('');
    try {
      const { data } = await axios.post<{ task_id: string; storyboard_id: string }>(
        '/api/video/storyboard/generate',
        { style, lora_names: loraNames, scenes }
      );
      setStoryboardId(data.storyboard_id);

      // Secondary progress poll: count how many scenes have at least one image
      imgProgressPollRef.current = setInterval(async () => {
        try {
          const { data: imgData } = await axios.get<StoryboardImagesResponse>(
            `/api/video/storyboard/${data.storyboard_id}/images`
          );
          const done = imgData.scenes.filter(s => s.versions.length > 0).length;
          setProgressSceneCount(done);
        } catch { /* keep polling */ }
      }, 3000);

      // Task completion poll
      genPollRef.current = setInterval(async () => {
        try {
          const { data: task } = await axios.get<{ status: string }>(`/api/tasks/${data.task_id}`);
          if (task.status === 'completed') {
            clearInterval(genPollRef.current!);
            genPollRef.current = null;
            clearInterval(imgProgressPollRef.current!);
            imgProgressPollRef.current = null;
            await fetchImages(data.storyboard_id);
            setPhase('ready');
          } else if (task.status === 'failed' || task.status === 'cancelled') {
            clearInterval(genPollRef.current!);
            genPollRef.current = null;
            clearInterval(imgProgressPollRef.current!);
            imgProgressPollRef.current = null;
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

  const regenerateScene = async (sceneIdx: number) => {
    if (!storyboardId || regenRunning[sceneIdx]) return;
    setRegenRunning(prev => ({ ...prev, [sceneIdx]: true }));

    // Collect current weights and prompt edit for this scene
    const activeScene_ = imagesData?.scenes.find(s => s.scene_idx === sceneIdx);
    const activeLoras = activeScene_ ? getActiveLorasForScene(activeScene_) : [];
    const weights = getSceneWeights(sceneIdx, activeLoras);
    const editedPrompt = promptEdits[sceneIdx]?.trim() || null;

    try {
      const { data } = await axios.post<{ task_id: string }>(
        `/api/video/storyboard/${storyboardId}/scene/${sceneIdx}/regenerate`,
        {
          positive_prompt: editedPrompt,
          seed: null,
          lora_weights: Object.keys(weights).length > 0 ? weights : null,
        }
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

  // Update a single LoRA weight for the active scene
  const setLoraWeight = (sceneIdx: number, loraName: string, weight: number) => {
    setSceneLoraWeights(prev => ({
      ...prev,
      [sceneIdx]: { ...(prev[sceneIdx] ?? {}), [loraName]: weight },
    }));
  };

  // Apply a preset to all active LoRAs for the current scene
  const applyPreset = (preset: 'style-priority' | 'balanced' | 'lora-priority') => {
    if (!imagesData) return;
    const sceneData = imagesData.scenes.find(s => s.scene_idx === activeScene);
    if (!sceneData) return;
    const activeLoras = getActiveLorasForScene(sceneData);
    const newWeights: Record<string, number> = {};
    activeLoras.forEach(l => {
      if (preset === 'style-priority') {
        newWeights[l.name] = l.type === 'character' ? 0.7 : 0.3;
      } else if (preset === 'balanced') {
        newWeights[l.name] = 0.6;
      } else {
        newWeights[l.name] = 0.9;
      }
    });
    setSceneLoraWeights(prev => ({
      ...prev,
      [activeScene]: { ...(prev[activeScene] ?? {}), ...newWeights },
    }));
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
    const pct = scenes.length > 0 ? Math.round((progressSceneCount / scenes.length) * 100) : 0;
    return (
      <div data-testid="stage-storyboard-generating" style={{ textAlign: 'center', padding: '40px 0' }}>
        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>🎨</div>
        <p style={{ fontSize: '1rem', marginBottom: '8px' }}>
          Generating {scenes.length} storyboard frames with SDXL…
        </p>
        {progressSceneCount > 0 && (
          <p style={{ color: '#aaa', fontSize: '0.88rem', marginBottom: '12px' }}>
            {progressSceneCount} of {scenes.length} scenes ready
          </p>
        )}
        <div style={{
          background: '#0f3460', borderRadius: '6px', height: '6px',
          width: '240px', margin: '0 auto 8px',
        }}>
          <div style={{
            background: '#e94560', height: '6px', borderRadius: '6px',
            width: `${pct}%`, transition: 'width 0.4s',
          }} />
        </div>
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

  // LoRA sliders for the active scene
  const activeSceneLoras = activeSceneData ? getActiveLorasForScene(activeSceneData) : [];
  const currentWeights = getSceneWeights(activeScene, activeSceneLoras);

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
              <span className="spin" style={{ color: '#e94560', fontSize: '1.4rem', marginRight: '8px' }}>⟳</span>
              <span style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>Generating…</span>
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

        {/* Right panel: scene info + version picker + regen + LoRA sliders */}
        <div style={{ width: '200px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '12px' }}>
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
                {activeSceneData.versions.map(v => {
                  const weightEntries = Object.entries(v.lora_weights ?? {});
                  const weightTip = weightEntries.length > 0
                    ? ` | Weights: ${weightEntries.map(([k, w]) => `${k}=${w}`).join(', ')}`
                    : '';
                  return (
                    <button
                      key={v.version}
                      data-testid={`version-btn-${activeScene}-${v.version}`}
                      onClick={() => setSelectedVersions(prev => ({ ...prev, [activeScene]: v.version }))}
                      title={`Seed: ${v.seed}${weightTip}`}
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
                  );
                })}
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
            data-testid="regenerate-scene-btn"
            onClick={() => void regenerateScene(activeScene)}
            disabled={isActiveRegen}
            style={{ fontSize: '0.8rem', background: '#0f3460' }}
          >
            {isActiveRegen ? <><span className="spin">⟳</span> Generating…</> : '↺ Regenerate Scene'}
          </button>

          {/* ── LoRA weight sliders ── */}
          {activeSceneLoras.length > 0 && (
            <div>
              <div style={{ fontSize: '0.7rem', color: '#666', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '6px' }}>
                LoRA Weights
              </div>
              {activeSceneLoras.map(l => {
                const w = currentWeights[l.name] ?? 0.7;
                return (
                  <div key={l.name} style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '2px' }}>
                      <span style={{ color: '#aaa' }}>{l.name}</span>
                      <span
                        data-testid={`lora-weight-value-${l.name}`}
                        style={{ color: '#e94560', minWidth: '28px', textAlign: 'right' }}
                      >
                        {w.toFixed(1)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={w}
                      data-testid={`lora-weight-slider-${l.name}`}
                      onChange={e => setLoraWeight(activeScene, l.name, parseFloat(e.target.value))}
                      style={{ width: '100%', accentColor: '#e94560' }}
                    />
                  </div>
                );
              })}

              {/* Weight presets */}
              <div style={{ display: 'flex', gap: '4px', marginTop: '4px', flexWrap: 'wrap' }}>
                <button
                  data-testid="preset-style-priority"
                  onClick={() => applyPreset('style-priority')}
                  title="Character LoRAs → 0.7 | Style LoRAs → 0.3"
                  style={{ fontSize: '0.65rem', padding: '2px 6px', background: '#0f3460' }}
                >
                  Style
                </button>
                <button
                  data-testid="preset-balanced"
                  onClick={() => applyPreset('balanced')}
                  title="All LoRAs → 0.6"
                  style={{ fontSize: '0.65rem', padding: '2px 6px', background: '#0f3460' }}
                >
                  Balanced
                </button>
                <button
                  data-testid="preset-lora-priority"
                  onClick={() => applyPreset('lora-priority')}
                  title="All LoRAs → 0.9"
                  style={{ fontSize: '0.65rem', padding: '2px 6px', background: '#0f3460' }}
                >
                  LoRA
                </button>
              </div>
            </div>
          )}

          {/* ── Prompt override ── */}
          {activeSceneData && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.7rem', color: '#666', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  Prompt
                </span>
                {promptEdits[activeScene] !== undefined && (
                  <button
                    data-testid="prompt-reset-btn"
                    onClick={() => setPromptEdits(prev => { const n = { ...prev }; delete n[activeScene]; return n; })}
                    title="Reset to original prompt"
                    style={{ fontSize: '0.6rem', padding: '1px 5px', background: '#16213e', color: '#888' }}
                  >
                    reset
                  </button>
                )}
              </div>
              <textarea
                data-testid="prompt-edit-textarea"
                value={promptEdits[activeScene] ?? activeSceneData.positive_prompt}
                onChange={e => setPromptEdits(prev => ({ ...prev, [activeScene]: e.target.value }))}
                rows={4}
                style={{
                  width: '100%',
                  fontSize: '0.7rem',
                  background: '#0a1a2e',
                  border: `1px solid ${promptEdits[activeScene] !== undefined ? '#e94560' : '#0f3460'}`,
                  borderRadius: '4px',
                  color: '#e0e0e0',
                  padding: '5px',
                  resize: 'vertical',
                  boxSizing: 'border-box',
                  lineHeight: 1.5,
                }}
              />
            </div>
          )}

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
                {isRegen ? <span className="spin">⟳</span> : `#${s.scene_idx + 1}`}
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
