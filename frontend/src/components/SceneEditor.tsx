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

function toMMSS(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
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
    <div data-testid="scene-editor">
      <h3 style={{ marginBottom: '12px' }}>Scene Editor ({scenes.length} scenes)</h3>
      <div data-testid="scene-list" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {scenes.map(scene => (
          <div
            key={scene.scene_index}
            data-testid={`scene-row-${scene.scene_index}`}
            style={{
              background: '#16213e',
              border: `1px solid ${scene.is_hero ? '#e94560' : '#0f3460'}`,
              borderRadius: '6px',
              padding: '10px 14px',
            }}
          >
            {/* Header row */}
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '8px', fontSize: '0.8rem' }}>
              <span style={{ fontWeight: 600, color: '#e0e0e0' }}>Scene {scene.scene_index + 1}</span>
              {scene.is_hero && (
                <span style={{ background: '#e94560', padding: '1px 6px', borderRadius: '3px', fontSize: '0.7rem' }}>★ hero</span>
              )}
              <span style={{ color: '#888' }}>{toMMSS(scene.start_sec)} – {toMMSS(scene.end_sec)}</span>
              <span style={{ color: '#888' }}>{scene.duration_sec.toFixed(1)}s</span>
              {scene.notes && <span style={{ color: '#666', textTransform: 'uppercase', fontSize: '0.7rem' }}>{scene.notes}</span>}
              {scene.backend_name && <span style={{ color: '#666' }}>{scene.backend_name}</span>}
            </div>

            {/* Prompt area */}
            {editingIndex === scene.scene_index ? (
              <div data-testid={`scene-edit-${scene.scene_index}`}>
                <textarea
                  value={draftPrompt}
                  onChange={e => setDraftPrompt(e.target.value)}
                  rows={4}
                  style={{ width: '100%', resize: 'vertical', fontSize: '0.82rem', fontFamily: 'inherit', marginBottom: '8px' }}
                  data-testid="scene-prompt-textarea"
                />
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button onClick={() => void saveEdit(scene.scene_index)} disabled={saving}>
                    {saving ? 'Saving…' : 'Save & Regenerate'}
                  </button>
                  <button onClick={cancelEdit} style={{ background: '#6c757d' }}>Cancel</button>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px' }}>
                <p style={{ margin: 0, fontSize: '0.82rem', color: scene.prompt ? '#e0e0e0' : '#555', fontStyle: scene.prompt ? 'normal' : 'italic', flex: 1 }}>
                  {scene.prompt ?? '(auto-generated)'}
                </p>
                <button
                  onClick={() => startEdit(scene)}
                  data-testid={`edit-scene-${scene.scene_index}-btn`}
                  style={{ padding: '3px 10px', fontSize: '0.75rem', flexShrink: 0 }}
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
