/**
 * VideoStudio — the main video creation workspace.
 * Multi-stage workflow: style → LoRA → plan → scene edit → generate → preview.
 */
import { useState } from 'react';
import axios from 'axios';
import { StyleSelector } from './StyleSelector';
import { LoRAManager } from './LoRAManager';
import { ExecutionPlanner } from './ExecutionPlanner';
import { SceneEditor } from './SceneEditor';
import { ProgressTracker } from './ProgressTracker';
import { VideoPreview } from './VideoPreview';
import type { ExecutionPlan, ScenePlan } from '../types';

type Stage = 'style' | 'lora' | 'plan' | 'scenes' | 'generating' | 'preview';

interface Props {
  audioId: string;
  songTitle?: string;
  onBack?: () => void;
}

export function VideoStudio({ audioId, songTitle, onBack }: Props) {
  const [stage, setStage] = useState<Stage>('style');
  const [selectedStyle, setSelectedStyle] = useState('cinematic');
  const [creativeDirection, setCreativeDirection] = useState('');
  const [executionPlan, setExecutionPlan] = useState<ExecutionPlan | null>(null);
  const [scenes, setScenes] = useState<ScenePlan[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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

  return (
    <div className="video-studio" data-testid="video-studio">
      <div className="studio-header">
        <h2>Video Studio{songTitle ? ` — ${songTitle}` : ''}</h2>
        {onBack && <button onClick={onBack} data-testid="back-btn">← Back</button>}
      </div>

      {/* Stage: Style selection */}
      {stage === 'style' && (
        <div data-testid="stage-style">
          <div className="creative-direction">
            <label>Creative direction (optional)</label>
            <textarea
              placeholder="Describe your vision… e.g. 'A neon-lit city at night, rain, vivid colors'"
              value={creativeDirection}
              onChange={e => setCreativeDirection(e.target.value)}
              data-testid="creative-direction-input"
            />
          </div>
          <StyleSelector
            selected={selectedStyle}
            onSelect={setSelectedStyle}
          />
          <button
            onClick={() => setStage('lora')}
            disabled={!selectedStyle}
            data-testid="next-to-lora-btn"
          >
            Next: LoRA Setup →
          </button>
        </div>
      )}

      {/* Stage: LoRA recommendations */}
      {stage === 'lora' && (
        <div data-testid="stage-lora">
          <LoRAManager audioId={audioId} style={selectedStyle} embedded />
          <div className="stage-nav">
            <button onClick={() => setStage('style')} data-testid="back-to-style-btn">← Style</button>
            <button onClick={() => void fetchPlan()} disabled={loading} data-testid="next-to-plan-btn">
              {loading ? 'Planning…' : 'Next: Execution Plan →'}
            </button>
          </div>
        </div>
      )}

      {/* Stage: Execution plan */}
      {stage === 'plan' && executionPlan && (
        <div data-testid="stage-plan">
          <ExecutionPlanner plan={executionPlan} onConfirm={() => setStage('scenes')} />
        </div>
      )}

      {/* Stage: Scene editor */}
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

      {/* Stage: Generating */}
      {stage === 'generating' && (
        <div data-testid="stage-generating">
          <ProgressTracker taskId={taskId} onComplete={handleGenerationComplete} />
        </div>
      )}

      {/* Stage: Preview */}
      {stage === 'preview' && videoId && (
        <div data-testid="stage-preview">
          <VideoPreview videoId={videoId} />
        </div>
      )}
    </div>
  );
}
