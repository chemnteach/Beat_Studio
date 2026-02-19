/**
 * CostEstimator — displays cost and time estimates for an execution plan.
 */
import type { ExecutionPlan } from '../types';

interface Props {
  plan: ExecutionPlan | null;
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function CostEstimator({ plan }: Props) {
  if (!plan) {
    return (
      <div className="cost-estimator empty" data-testid="cost-estimator">
        <p>No execution plan yet. Select a style and continue to see cost estimates.</p>
      </div>
    );
  }

  return (
    <div className="cost-estimator" data-testid="cost-estimator">
      <h4>Cost Estimate</h4>
      <table className="cost-table" data-testid="cost-table">
        <tbody>
          <tr>
            <td>Backend</td>
            <td data-testid="est-backend">{plan.backend}</td>
          </tr>
          <tr>
            <td>Scenes</td>
            <td data-testid="est-scenes">{plan.num_scenes}</td>
          </tr>
          <tr>
            <td>Estimated Time</td>
            <td data-testid="est-time">{formatTime(plan.estimated_time_sec)}</td>
          </tr>
          <tr>
            <td>Estimated Cost</td>
            <td data-testid="est-cost">
              {plan.is_local
                ? <span className="free">Free (local GPU)</span>
                : <span className="paid">${plan.estimated_cost_usd.toFixed(2)}</span>
              }
            </td>
          </tr>
          <tr>
            <td>Mode</td>
            <td>
              <span
                className={`mode-badge ${plan.is_local ? 'local' : 'cloud'}`}
                data-testid="est-mode"
              >
                {plan.is_local ? 'Local' : 'Cloud'}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
