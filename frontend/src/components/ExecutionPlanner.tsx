/**
 * ExecutionPlanner — presents local vs cloud execution options with cost/time.
 * The user selects a path before generation begins.
 * Supports a "test run" mode: pick specific scenes via checkboxes.
 * The "first N" input is a quick-select shortcut; individual boxes can be toggled.
 */
import { useState } from 'react';
import type { ExecutionPlan } from '../types';

interface Props {
  plan: ExecutionPlan;
  onConfirm: (planId: string, sceneIndices?: number[]) => void;
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function toMMSS(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function ExecutionPlanner({ plan, onConfirm }: Props) {
  const [testMode, setTestMode] = useState(false);
  const [testCount, setTestCount] = useState(Math.min(4, plan.num_scenes));
  const [selected, setSelected] = useState<Set<number>>(() => new Set());

  // When test mode turns on, pre-select the first testCount scenes
  const handleTestModeToggle = (on: boolean) => {
    setTestMode(on);
    if (on) preselectFirst(testCount);
  };

  const preselectFirst = (n: number) => {
    const indices = (plan.scenes ?? []).slice(0, n).map(s => s.scene_index);
    // If plan.scenes is empty, fall back to 0..n-1
    const fallback = indices.length ? indices : Array.from({ length: n }, (_, i) => i);
    setSelected(new Set(fallback));
  };

  const handleCountChange = (n: number) => {
    const clamped = Math.max(1, Math.min(plan.num_scenes, n));
    setTestCount(clamped);
    preselectFirst(clamped);
  };

  const toggleScene = (idx: number) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  };

  const selectedCount = selected.size;
  const estTime = testMode
    ? Math.round((plan.estimated_time_sec / Math.max(plan.num_scenes, 1)) * selectedCount)
    : plan.estimated_time_sec;

  const sceneIndices = testMode ? Array.from(selected).sort((a, b) => a - b) : undefined;

  return (
    <div data-testid="execution-planner" style={{ maxWidth: '560px' }}>
      <h3 style={{ marginBottom: '16px' }}>Execution Plan</h3>

      {/* Summary stats */}
      <div
        data-testid="plan-summary"
        style={{
          background: '#16213e', border: '1px solid #0f3460',
          borderRadius: '8px', overflow: 'hidden', marginBottom: '16px',
        }}
      >
        {[
          { label: 'Backend',        value: plan.backend,  testid: 'plan-backend' },
          { label: 'Scenes',         value: testMode ? `${selectedCount} of ${plan.num_scenes} (test)` : String(plan.num_scenes), testid: 'plan-scenes' },
          { label: 'Estimated Time', value: formatTime(estTime), testid: 'plan-time' },
          { label: 'Estimated Cost', value: plan.is_local ? 'Free (local GPU)' : `$${plan.estimated_cost_usd.toFixed(2)}`, testid: 'plan-cost' },
          { label: 'Mode',           value: plan.is_local ? 'Local' : 'Cloud', testid: 'plan-mode' },
        ].map(({ label, value, testid }, i) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 16px', borderBottom: i < 4 ? '1px solid #0f3460' : undefined }}>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>{label}</span>
            <span data-testid={testid} style={{ fontWeight: 600, color: '#e0e0e0' }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Test run section */}
      <div style={{
        padding: '12px 14px', marginBottom: '16px',
        background: testMode ? '#0a1f0a' : '#16213e',
        border: `1px solid ${testMode ? '#2ecc71' : '#0f3460'}`,
        borderRadius: '6px',
      }}>
        {/* Toggle + quick-select row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input
            type="checkbox"
            id="test-mode-toggle"
            checked={testMode}
            onChange={e => handleTestModeToggle(e.target.checked)}
            style={{ cursor: 'pointer', width: '16px', height: '16px', flexShrink: 0 }}
          />
          <label htmlFor="test-mode-toggle" style={{ cursor: 'pointer', fontSize: '0.85rem', color: testMode ? '#2ecc71' : '#aaa' }}>
            Test run — pre-select first
          </label>
          <input
            type="number"
            min={1}
            max={plan.num_scenes}
            value={testCount}
            onChange={e => handleCountChange(Number(e.target.value))}
            disabled={!testMode}
            style={{
              width: '48px', padding: '2px 6px', fontSize: '0.85rem',
              background: '#0f3460', color: '#e0e0e0', border: '1px solid #1a4a80',
              borderRadius: '4px', textAlign: 'center',
              opacity: testMode ? 1 : 0.4,
            }}
          />
          <span style={{ fontSize: '0.85rem', color: testMode ? '#2ecc71' : '#aaa' }}>
            scenes
          </span>
        </div>

        {/* Per-scene checkboxes (only shown in test mode with scene data) */}
        {testMode && plan.scenes && plan.scenes.length > 0 && (
          <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px', maxHeight: '240px', overflowY: 'auto' }}>
            {plan.scenes.map(scene => {
              const checked = selected.has(scene.scene_index);
              return (
                <label
                  key={scene.scene_index}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '4px 6px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.78rem',
                    background: checked ? '#0d2d0d' : '#0f1f2e',
                    border: `1px solid ${checked ? '#2ecc71' : '#0f3460'}`,
                    color: checked ? '#2ecc71' : '#888',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleScene(scene.scene_index)}
                    style={{ cursor: 'pointer', flexShrink: 0 }}
                  />
                  <span style={{ fontWeight: 600, color: checked ? '#e0e0e0' : '#aaa' }}>
                    {scene.scene_index + 1}
                  </span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {scene.notes ?? ''} {scene.start_sec != null ? toMMSS(scene.start_sec) : ''}
                    {scene.is_hero ? ' ★' : ''}
                  </span>
                </label>
              );
            })}
          </div>
        )}

        {testMode && selectedCount === 0 && (
          <p style={{ marginTop: '8px', fontSize: '0.8rem', color: '#e74c3c' }}>Select at least one scene.</p>
        )}
      </div>

      {!plan.is_local && (
        <div
          data-testid="cloud-warning"
          style={{ padding: '10px 14px', background: '#1a1010', border: '1px solid #c0392b', borderRadius: '6px', fontSize: '0.83rem', color: '#e74c3c', marginBottom: '16px' }}
        >
          This plan uses cloud rendering. Costs will be billed to your RunPod account.
        </div>
      )}

      <button
        data-testid="confirm-plan-btn"
        onClick={() => onConfirm(plan.plan_id, sceneIndices)}
        disabled={testMode && selectedCount === 0}
        style={{ padding: '10px 24px', fontSize: '1rem', background: testMode ? '#2ecc71' : '#e94560', color: '#111', opacity: testMode && selectedCount === 0 ? 0.5 : 1 }}
      >
        {testMode ? `Generate Test (${selectedCount} scene${selectedCount !== 1 ? 's' : ''}) →` : 'Confirm and Generate →'}
      </button>
    </div>
  );
}
