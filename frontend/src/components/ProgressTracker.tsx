/**
 * ProgressTracker — real-time generation progress with WebSocket + polling fallback.
 * Shows stage-by-stage progress bars and estimated time remaining.
 */
import { useWebSocket } from '../hooks/useWebSocket';
import { useTaskPolling } from '../hooks/useTaskPolling';

interface Props {
  taskId: string | null;
  onComplete: () => void;
}

export function ProgressTracker({ taskId, onComplete }: Props) {
  const ws = useWebSocket(taskId);
  const poll = useTaskPolling(ws.connected ? null : taskId);  // fallback only

  // Use WebSocket if connected, otherwise polling
  const progress = ws.connected ? ws.progress : (poll.task?.progress ?? 0);
  const stage = ws.connected ? ws.stage : (poll.task?.stage ?? '');
  const isComplete = (poll.task?.status === 'completed') || progress >= 100;

  if (isComplete && onComplete) {
    onComplete();
  }

  return (
    <div className="progress-tracker" data-testid="progress-tracker">
      <h3>Generating Video…</h3>

      {stage && (
        <p className="current-stage" data-testid="current-stage">{stage}</p>
      )}

      <div className="progress-bar-container" data-testid="progress-bar-container">
        <div
          className="progress-bar"
          style={{ width: `${Math.min(progress, 100)}%` }}
          data-testid="progress-bar"
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      <p className="progress-pct" data-testid="progress-pct">{Math.round(progress)}%</p>

      <div className="connection-status" data-testid="connection-status">
        {ws.connected
          ? <span className="live">Live updates</span>
          : <span className="polling">Polling…</span>
        }
      </div>

      {poll.task?.status === 'failed' && (
        <div className="error" data-testid="generation-error">
          Generation failed: {poll.task.error ?? 'Unknown error'}
        </div>
      )}
    </div>
  );
}
