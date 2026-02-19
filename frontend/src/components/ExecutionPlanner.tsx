/**
 * ExecutionPlanner — presents local vs cloud execution options with cost/time.
 * The user selects a path before generation begins.
 */
import type { ExecutionPlan } from '../types';

interface Props {
  plan: ExecutionPlan;
  onConfirm: (planId: string) => void;
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function ExecutionPlanner({ plan, onConfirm }: Props) {
  return (
    <div className="execution-planner" data-testid="execution-planner">
      <h3>Execution Plan</h3>

      <div className="plan-summary" data-testid="plan-summary">
        <div className="plan-stat">
          <span className="label">Backend</span>
          <span className="value" data-testid="plan-backend">{plan.backend}</span>
        </div>
        <div className="plan-stat">
          <span className="label">Scenes</span>
          <span className="value" data-testid="plan-scenes">{plan.num_scenes}</span>
        </div>
        <div className="plan-stat">
          <span className="label">Estimated Time</span>
          <span className="value" data-testid="plan-time">{formatTime(plan.estimated_time_sec)}</span>
        </div>
        <div className="plan-stat">
          <span className="label">Estimated Cost</span>
          <span className="value" data-testid="plan-cost">
            {plan.is_local ? 'Free (local GPU)' : `$${plan.estimated_cost_usd.toFixed(2)}`}
          </span>
        </div>
        <div className="plan-stat">
          <span className="label">Mode</span>
          <span className={`value ${plan.is_local ? 'local' : 'cloud'}`} data-testid="plan-mode">
            {plan.is_local ? 'Local' : 'Cloud'}
          </span>
        </div>
      </div>

      {!plan.is_local && (
        <div className="cloud-warning" data-testid="cloud-warning">
          This plan uses cloud rendering. Costs will be billed to your RunPod account.
        </div>
      )}

      <button
        className="confirm-btn"
        onClick={() => onConfirm(plan.plan_id)}
        data-testid="confirm-plan-btn"
      >
        Confirm and Generate
      </button>
    </div>
  );
}
