/**
 * SceneEditor — per-scene prompt review and editing.
 * Lists all scenes with auto-generated prompts; allows editing any scene.
 */
import { useState } from 'react';
import axios from 'axios';
import type { ScenePlan } from '../types';

interface Props {
  videoId: string;
  scenes: ScenePlan[];
  onSceneUpdated: (sceneIndex: number) => void;
}

export function SceneEditor({ videoId, scenes, onSceneUpdated }: Props) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [draftPrompt, setDraftPrompt] = useState('');
  const [saving, setSaving] = useState(false);

  const startEdit = (scene: ScenePlan) => {
    setEditingIndex(scene.scene_index);
    setDraftPrompt(scene.prompt ?? '');
  };

  const cancelEdit = () => {
    setEditingIndex(null);
    setDraftPrompt('');
  };

  const saveEdit = async (sceneIndex: number) => {
    setSaving(true);
    try {
      await axios.post('/api/video/scene/edit', {
        video_id: videoId,
        scene_index: sceneIndex,
        new_prompt: draftPrompt,
      });
      onSceneUpdated(sceneIndex);
      cancelEdit();
    } catch { /* non-fatal */ } finally {
      setSaving(false);
    }
  };

  return (
    <div className="scene-editor" data-testid="scene-editor">
      <h3>Scene Editor ({scenes.length} scenes)</h3>
      <div className="scene-list" data-testid="scene-list">
        {scenes.map(scene => (
          <div key={scene.scene_index} className="scene-row" data-testid={`scene-row-${scene.scene_index}`}>
            <div className="scene-header">
              <span className="scene-index">Scene {scene.scene_index + 1}</span>
              <span className="scene-backend">{scene.backend_name}</span>
              <span className="scene-time">{scene.estimated_time_sec.toFixed(0)}s est.</span>
              {scene.estimated_cost_usd > 0 && (
                <span className="scene-cost">${scene.estimated_cost_usd.toFixed(4)}</span>
              )}
            </div>

            {editingIndex === scene.scene_index ? (
              <div className="scene-edit-form" data-testid={`scene-edit-${scene.scene_index}`}>
                <textarea
                  value={draftPrompt}
                  onChange={e => setDraftPrompt(e.target.value)}
                  rows={4}
                  data-testid="scene-prompt-textarea"
                />
                <div className="edit-actions">
                  <button onClick={() => void saveEdit(scene.scene_index)} disabled={saving}>
                    {saving ? 'Saving…' : 'Save & Regenerate'}
                  </button>
                  <button onClick={cancelEdit}>Cancel</button>
                </div>
              </div>
            ) : (
              <div className="scene-prompt-preview">
                <p>{scene.prompt ?? '(auto-generated)'}</p>
                <button
                  onClick={() => startEdit(scene)}
                  data-testid={`edit-scene-${scene.scene_index}-btn`}
                >
                  Edit
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
