/**
 * VideoStudio — the main video creation workspace.
 * Stage flow: style → prompts → lora → plan → scenes → generating → preview
 */
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { StyleSelector } from './StyleSelector';
import { LoRAManager } from './LoRAManager';
import { ExecutionPlanner } from './ExecutionPlanner';
import { SceneEditor } from './SceneEditor';
import { ProgressTracker } from './ProgressTracker';
import { VideoPreview } from './VideoPreview';
import type { ExecutionPlan, ScenePlan, SectionInfo, SongAnalysis } from '../types';

type Stage = 'style' | 'prompts' | 'lora' | 'plan' | 'scenes' | 'generating' | 'preview';

interface ScenePrompt {
  scene_index: number;
  section_type: string;
  start_sec: number;
  end_sec: number;
  is_hero: boolean;
  positive: string;
  negative: string;
  transition: string;
}

interface PromptsResult {
  overall_concept: string;
  color_palette: string[];
  mood_progression: string;
  prompts: ScenePrompt[];
}

interface BackendInfo {
  name: string;
  available: boolean;
  is_local: boolean;
  vram_required_gb: number;
  cost_per_scene: number;
}

const BACKEND_LABELS: Record<string, { label: string; note: string }> = {
  animatediff:     { label: 'AnimateDiff',        note: '5.6 GB VRAM — fast, animated styles' },
  sdxl_controlnet: { label: 'SDXL ControlNet',    note: '8 GB VRAM — rotoscope / structure' },
  svd:             { label: 'SVD',                 note: '10 GB VRAM — image-to-video' },
  wan26_local:     { label: 'WAN 2.6 Local',       note: '16 GB VRAM — highest local quality' },
  wan26_cloud:     { label: 'WAN 2.6 Cloud',       note: 'RunPod — best quality, ~$0.05/scene' },
  skyreels:        { label: 'SkyReels V2',         note: 'RunPod — seamless stitching (DF)' },
  cogvideox:       { label: 'CogVideoX',           note: 'Cloud — stub' },
  mochi:           { label: 'Mochi',               note: 'Cloud — stub' },
  ltx_video:       { label: 'LTX Video',           note: 'Cloud — stub' },
};

interface Props {
  audioId: string;
  analysis?: SongAnalysis;
  sections?: SectionInfo[];
  onBack?: () => void;
}

function toMMSS(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function VideoStudio({ audioId, analysis, sections, onBack }: Props) {
  const [stage, setStage] = useState<Stage>('style');
  const [selectedStyle, setSelectedStyle] = useState('cinematic');
  const [selectedBackend, setSelectedBackend] = useState('animatediff');
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [showBackends, setShowBackends] = useState(false);
  const [creativeDirection, setCreativeDirection] = useState('');
  const [promptsResult, setPromptsResult] = useState<PromptsResult | null>(null);
  const [editedPrompts, setEditedPrompts] = useState<ScenePrompt[]>([]);
  const [loadingPrompts, setLoadingPrompts] = useState(false);
  const [promptsError, setPromptsError] = useState('');
  const [executionPlan, setExecutionPlan] = useState<ExecutionPlan | null>(null);
  const [scenes, setScenes] = useState<ScenePlan[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadNotice, setUploadNotice] = useState('');
  const [sceneIndices, setSceneIndices] = useState<number[] | undefined>(undefined);
  const [selectedLoraNames, setSelectedLoraNames] = useState<string[]>([]);

  const songTitle = analysis?.title;

  // Load backend list once on mount
  useEffect(() => {
    axios.get<{ backends: BackendInfo[] }>('/api/video/backends')
      .then(({ data }) => setBackends(data.backends))
      .catch(() => {});
  }, []);

  const handleStyleSelect = (style: string, recommendedBackend?: string) => {
    setSelectedStyle(style);
    if (recommendedBackend && backends.some(b => b.name === recommendedBackend)) {
      setSelectedBackend(recommendedBackend);
    }
  };

  const generatePrompts = async () => {
    setLoadingPrompts(true);
    setPromptsError('');
    try {
      const payload: Record<string, unknown> = {
        audio_id: audioId,
        style: selectedStyle,
        quality: 'professional',
        creative_direction: creativeDirection,
      };
      // Send user-edited sections (with lyrical content) if available
      if (sections && sections.length > 0) {
        payload.sections = sections.map(s => ({
          section_type: s.section_type,
          start_sec: s.start_sec,
          end_sec: s.end_sec,
          lyrical_content: s.lyrical_content ?? '',
        }));
      }
      const { data } = await axios.post<PromptsResult>('/api/video/prompts', payload);
      setPromptsResult(data);
      setEditedPrompts(data.prompts);
      setStage('prompts');
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Prompt generation failed';
      setPromptsError(msg);
    } finally {
      setLoadingPrompts(false);
    }
  };

  const fetchPlan = async () => {
    setLoading(true);
    try {
      const selectedBackendInfo = backends.find(b => b.name === selectedBackend);
      const { data } = await axios.post<ExecutionPlan>('/api/video/plan', {
        audio_id: audioId,
        style: selectedStyle,
        quality: 'professional',
        local_preferred: selectedBackendInfo?.is_local ?? true,
      });
      setExecutionPlan(data);
      setScenes(data.scenes ?? []);
      setStage('plan');
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  const startGeneration = async (planId: string, indices?: number[]) => {
    try {
      // Build user_overrides from the reviewed prompts stage (scene_index → positive)
      const user_overrides: Record<string, string> = {};
      for (const p of editedPrompts) {
        if (p.positive?.trim()) {
          user_overrides[String(p.scene_index)] = p.positive.trim();
        }
      }
      const body: Record<string, unknown> = {
        plan_id: planId,
        audio_id: audioId,
        style: selectedStyle,
        quality: 'professional',
        creative_direction: creativeDirection,
        user_overrides,
        lora_names: selectedLoraNames,
      };
      if (indices && indices.length > 0) body.scene_indices = indices;
      const { data } = await axios.post<{ task_id: string }>('/api/video/generate', body);
      setTaskId(data.task_id);
      setStage('generating');
    } catch { /* non-fatal */ }
  };

  const handleGenerationComplete = async () => {
    if (taskId) {
      try {
        const { data } = await axios.get<{ result?: { video_id?: string } }>(`/api/tasks/${taskId}`);
        const realVideoId = data.result?.video_id ?? taskId;
        setVideoId(realVideoId);
      } catch {
        setVideoId(taskId);
      }
      setStage('preview');
    }
  };

  const updatePrompt = (i: number, positive: string) => {
    setEditedPrompts(prev => prev.map((p, idx) => idx === i ? { ...p, positive } : p));
  };

  const promptsFileRef = useRef<HTMLInputElement>(null);

  const downloadPromptsJson = () => {
    const payload = {
      version: 1,
      audio_id: audioId,
      style: selectedStyle,
      backend: selectedBackend,
      creative_direction: creativeDirection,
      overall_concept: promptsResult?.overall_concept ?? '',
      color_palette: promptsResult?.color_palette ?? [],
      mood_progression: promptsResult?.mood_progression ?? '',
      prompts: editedPrompts,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prompts_${audioId}_${selectedStyle}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePromptsUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        if (Array.isArray(data.prompts)) {
          setEditedPrompts(data.prompts);
          setUploadNotice(`✓ Loaded ${data.prompts.length} prompts from "${file.name}"`);
          setTimeout(() => setUploadNotice(''), 4000);
        }
        if (data.overall_concept !== undefined || data.color_palette !== undefined || data.mood_progression !== undefined) {
          setPromptsResult({
            overall_concept: data.overall_concept ?? promptsResult?.overall_concept ?? '',
            color_palette: data.color_palette ?? promptsResult?.color_palette ?? [],
            mood_progression: data.mood_progression ?? promptsResult?.mood_progression ?? '',
            prompts: data.prompts ?? editedPrompts,
          });
        }
        if (data.creative_direction) setCreativeDirection(data.creative_direction);
        if (data.style) setSelectedStyle(data.style as string);
        if (data.backend) setSelectedBackend(data.backend as string);
      } catch { /* invalid JSON — ignore */ }
    };
    reader.readAsText(file);
    // Reset so the same file can be re-uploaded after edits
    e.target.value = '';
  };

  return (
    <div className="video-studio" data-testid="video-studio">
      <div className="studio-header">
        <h2>Video Studio{songTitle ? ` — ${songTitle}` : ''}</h2>
        {onBack && <button onClick={onBack} data-testid="back-btn">← Back</button>}
      </div>

      {/* ── Stage: Style + Creative direction ── */}
      {stage === 'style' && (
        <div data-testid="stage-style">
          <div className="creative-direction" style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '6px', color: '#aaa', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
              Artist / Creative Direction
            </label>
            <textarea
              placeholder="Paste the artist's vision, notes, or inspiration here… e.g. 'Sun-drenched tropical beach bar, golden hour, searching for someone special, bittersweet longing'"
              value={creativeDirection}
              onChange={e => setCreativeDirection(e.target.value)}
              rows={4}
              style={{ width: '100%', resize: 'vertical' }}
              data-testid="creative-direction-input"
            />
          </div>

          {sections && sections.length > 0 && (
            <div style={{ marginBottom: '12px', padding: '8px 12px', background: '#16213e', borderRadius: '6px', fontSize: '0.8rem', color: '#888' }}>
              ✓ {sections.length} sections with lyrical content will be used for prompt generation
            </div>
          )}

          <StyleSelector selected={selectedStyle} onSelect={handleStyleSelect} />

          {/* Backend picker */}
          <div style={{ margin: '16px 0' }}>
            <button
              onClick={() => setShowBackends(v => !v)}
              style={{ background: '#0f3460', fontSize: '0.8rem', padding: '4px 12px' }}
            >
              {showBackends ? '▲' : '▼'} Generation Backend: <strong style={{ color: '#e94560' }}>{BACKEND_LABELS[selectedBackend]?.label ?? selectedBackend}</strong>
            </button>

            {showBackends && (
              <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {['Local (your GPU)', 'Cloud (RunPod / API)'].map(group => {
                  const isLocal = group.startsWith('Local');
                  const groupBackends = backends.filter(b => b.is_local === isLocal);
                  if (groupBackends.length === 0) return null;
                  return (
                    <div key={group}>
                      <div style={{ fontSize: '0.7rem', color: '#888', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '1px' }}>{group}</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {groupBackends.map(b => {
                          const meta = BACKEND_LABELS[b.name] ?? { label: b.name, note: '' };
                          const isSelected = selectedBackend === b.name;
                          return (
                            <button
                              key={b.name}
                              onClick={() => { setSelectedBackend(b.name); setShowBackends(false); }}
                              title={meta.note}
                              style={{
                                padding: '5px 12px',
                                fontSize: '0.78rem',
                                background: isSelected ? '#e94560' : b.available ? '#16213e' : '#0a0a1a',
                                border: `1px solid ${isSelected ? '#e94560' : b.available ? '#0f3460' : '#222'}`,
                                color: b.available ? '#e0e0e0' : '#555',
                                borderRadius: '6px',
                                cursor: 'pointer',
                              }}
                            >
                              {meta.label}
                              {!b.available && <span style={{ fontSize: '0.65rem', marginLeft: '4px', color: '#666' }}>(not installed)</span>}
                              {b.available && isLocal && <span style={{ fontSize: '0.65rem', marginLeft: '4px', color: '#888' }}>{b.vram_required_gb}GB</span>}
                              {b.available && !isLocal && b.cost_per_scene > 0 && <span style={{ fontSize: '0.65rem', marginLeft: '4px', color: '#888' }}>${b.cost_per_scene}/scene</span>}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {selectedStyle === 'rotoscope' && (
            <div style={{ marginTop: '12px', padding: '10px 14px', background: '#2a1010', border: '1px solid #c0392b', borderRadius: '6px', fontSize: '0.82rem', color: '#e74c3c' }}>
              <strong>Rotoscope requires source footage</strong> — this style needs a ControlNet input video (traced from live footage) and is not yet supported without one. Pick a different style to continue.
            </div>
          )}

          {selectedStyle === 'abstract' && (
            <div style={{ marginTop: '12px', padding: '10px 14px', background: '#1a1a10', border: '1px solid #b7950b', borderRadius: '6px', fontSize: '0.82rem', color: '#f1c40f' }}>
              <strong>Abstract suppresses figurative content</strong> — this style adds "non-representational" to the prompt and negates "figurative". Character LoRAs (e.g. rob_char) may not render effectively.
            </div>
          )}

          {promptsError && <p className="error">{promptsError}</p>}

          <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button
              onClick={() => void generatePrompts()}
              disabled={loadingPrompts || !selectedStyle || selectedStyle === 'rotoscope'}
              data-testid="generate-prompts-btn"
            >
              {loadingPrompts ? 'Generating prompts…' : 'Generate Prompts →'}
            </button>
            {loadingPrompts && (
              <span style={{ fontSize: '0.8rem', color: '#888' }}>
                Analyzing narrative arc with AI (this takes ~10s)…
              </span>
            )}
          </div>
        </div>
      )}

      {/* ── Stage: Review + edit prompts ── */}
      {stage === 'prompts' && promptsResult && (
        <div data-testid="stage-prompts">
          <div style={{ marginBottom: '16px', padding: '12px 16px', background: '#16213e', borderRadius: '8px', borderLeft: '3px solid #e94560' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{promptsResult.overall_concept}</div>
            <div style={{ fontSize: '0.8rem', color: '#888' }}>{promptsResult.mood_progression}</div>
            {promptsResult.color_palette?.length > 0 && (
              <div style={{ marginTop: '6px', display: 'flex', gap: '6px' }}>
                {promptsResult.color_palette.map((c, i) => (
                  <span key={i} style={{ fontSize: '0.75rem', background: '#0f3460', padding: '2px 8px', borderRadius: '12px', color: '#e0e0e0' }}>{c}</span>
                ))}
              </div>
            )}
          </div>

          <h3 style={{ marginBottom: '10px' }}>Scene Prompts <span style={{ fontSize: '0.75rem', color: '#888', fontWeight: 'normal' }}>(click any prompt to edit)</span></h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '20px' }}>
            {editedPrompts.map((p, i) => (
              <div
                key={i}
                style={{ background: '#16213e', border: `1px solid ${p.is_hero ? '#e94560' : '#0f3460'}`, borderRadius: '8px', padding: '12px' }}
              >
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ background: p.is_hero ? '#e94560' : '#0f3460', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', textTransform: 'uppercase' }}>
                    {p.section_type}{p.is_hero ? ' ★' : ''}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: '#888' }}>
                    {toMMSS(p.start_sec)} – {toMMSS(p.end_sec)}
                  </span>
                  <span style={{ fontSize: '0.7rem', color: '#666' }}>→ {p.transition}</span>
                </div>
                <textarea
                  value={p.positive}
                  onChange={e => updatePrompt(i, e.target.value)}
                  rows={2}
                  style={{ width: '100%', resize: 'vertical', fontSize: '0.82rem', fontFamily: 'inherit' }}
                />
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
            <button onClick={() => setStage('style')} style={{ background: '#6c757d' }}>
              ← Back to Style
            </button>
            <button onClick={downloadPromptsJson} style={{ background: '#0f3460' }} data-testid="download-prompts-btn">
              ↓ Download JSON
            </button>
            <button onClick={() => promptsFileRef.current?.click()} style={{ background: '#0f3460' }} data-testid="upload-prompts-btn">
              ↑ Upload JSON
            </button>
            <input
              ref={promptsFileRef}
              type="file"
              accept=".json,application/json"
              style={{ display: 'none' }}
              onChange={handlePromptsUpload}
            />
            {uploadNotice && (
              <span style={{ fontSize: '0.8rem', color: '#2ecc71' }}>{uploadNotice}</span>
            )}
            <button onClick={() => setStage('lora')} data-testid="next-to-lora-btn" style={{ marginLeft: 'auto' }}>
              Continue to LoRA Setup →
            </button>
          </div>
        </div>
      )}

      {/* ── Stage: LoRA recommendations ── */}
      {stage === 'lora' && (
        <div data-testid="stage-lora">
          <LoRAManager
            audioId={audioId}
            style={selectedStyle}
            embedded
            selectedNames={selectedLoraNames}
            onSelectionChange={setSelectedLoraNames}
          />
          <div className="stage-nav">
            <button onClick={() => setStage('prompts')} data-testid="back-to-prompts-btn">← Prompts</button>
            <button onClick={() => void fetchPlan()} disabled={loading} data-testid="next-to-plan-btn">
              {loading ? 'Planning…' : 'Next: Execution Plan →'}
            </button>
          </div>
        </div>
      )}

      {/* ── Stage: Execution plan ── */}
      {stage === 'plan' && executionPlan && (
        <div data-testid="stage-plan">
          <ExecutionPlanner plan={executionPlan} onConfirm={(planId, indices) => { setSceneIndices(indices); setStage('scenes'); }} />
        </div>
      )}

      {/* ── Stage: Scene editor ── */}
      {stage === 'scenes' && executionPlan && (
        <div data-testid="stage-scenes">
          {sceneIndices && (
            <div style={{ padding: '8px 14px', marginBottom: '12px', background: '#0a1f0a', border: '1px solid #2ecc71', borderRadius: '6px', fontSize: '0.82rem', color: '#2ecc71' }}>
              Test run — generating {sceneIndices.length} of {executionPlan.num_scenes} scenes: {sceneIndices.map(i => i + 1).join(', ')}
            </div>
          )}
          <SceneEditor
            videoId={executionPlan.plan_id}
            scenes={scenes}
            onSceneUpdated={idx => console.log('Scene updated:', idx)}
          />
          <div className="stage-nav">
            <button onClick={() => setStage('plan')} data-testid="back-to-plan-btn">← Plan</button>
            <button
              onClick={() => void startGeneration(executionPlan.plan_id, sceneIndices)}
              data-testid="start-generation-btn"
            >
              Generate Video →
            </button>
          </div>
        </div>
      )}

      {/* ── Stage: Generating ── */}
      {stage === 'generating' && (
        <div data-testid="stage-generating">
          <ProgressTracker taskId={taskId} onComplete={handleGenerationComplete} />
        </div>
      )}

      {/* ── Stage: Preview ── */}
      {stage === 'preview' && videoId && (
        <div data-testid="stage-preview">
          <VideoPreview videoId={videoId} videoUrl={`/api/video/download/${videoId}`} />
        </div>
      )}
    </div>
  );
}
