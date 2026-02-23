/**
 * VideoStudio — the main video creation workspace.
 * Stage flow: style → prompts → lora → plan → scenes → generating → preview
 */
import { useState } from 'react';
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

  const songTitle = analysis?.title;

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
      const { data } = await axios.post<ExecutionPlan>('/api/video/plan', {
        audio_id: audioId,
        style: selectedStyle,
        quality: 'professional',
        local_preferred: true,
      });
      setExecutionPlan(data);
      setScenes(data.scenes ?? []);
      setStage('plan');
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  const startGeneration = async (planId: string) => {
    try {
      const { data } = await axios.post<{ task_id: string }>('/api/video/generate', {
        plan_id: planId,
        audio_id: audioId,
      });
      setTaskId(data.task_id);
      setStage('generating');
    } catch { /* non-fatal */ }
  };

  const handleGenerationComplete = () => {
    if (taskId) {
      setVideoId(taskId);
      setStage('preview');
    }
  };

  const updatePrompt = (i: number, positive: string) => {
    setEditedPrompts(prev => prev.map((p, idx) => idx === i ? { ...p, positive } : p));
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

          <StyleSelector selected={selectedStyle} onSelect={setSelectedStyle} />

          {promptsError && <p className="error">{promptsError}</p>}

          <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button
              onClick={() => void generatePrompts()}
              disabled={loadingPrompts || !selectedStyle}
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

          <div style={{ display: 'flex', gap: '12px' }}>
            <button onClick={() => setStage('style')} style={{ background: '#6c757d' }}>
              ← Back to Style
            </button>
            <button onClick={() => void fetchPlan()} disabled={loading} data-testid="next-to-plan-btn">
              {loading ? 'Planning…' : 'Continue to LoRA Setup →'}
            </button>
          </div>
        </div>
      )}

      {/* ── Stage: LoRA recommendations ── */}
      {stage === 'lora' && (
        <div data-testid="stage-lora">
          <LoRAManager audioId={audioId} style={selectedStyle} embedded />
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
          <ExecutionPlanner plan={executionPlan} onConfirm={() => setStage('scenes')} />
        </div>
      )}

      {/* ── Stage: Scene editor ── */}
      {stage === 'scenes' && executionPlan && (
        <div data-testid="stage-scenes">
          <SceneEditor
            videoId={executionPlan.plan_id}
            scenes={scenes}
            onSceneUpdated={idx => console.log('Scene updated:', idx)}
          />
          <div className="stage-nav">
            <button onClick={() => setStage('plan')} data-testid="back-to-plan-btn">← Plan</button>
            <button
              onClick={() => void startGeneration(executionPlan.plan_id)}
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
          <VideoPreview videoId={videoId} />
        </div>
      )}
    </div>
  );
}
