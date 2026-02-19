/**
 * useTaskPolling — polling fallback for task progress when WebSocket is unavailable.
 *
 * Polls GET /api/tasks/{taskId} every intervalMs milliseconds.
 * Stops automatically when status is completed, failed, or cancelled.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';
import type { TaskState, TaskStatus } from '../types';

const TERMINAL_STATES: TaskStatus[] = ['completed', 'failed', 'cancelled'];
const DEFAULT_INTERVAL = 2000;

export interface PollingState {
  task: TaskState | null;
  loading: boolean;
  error: string | null;
  isTerminal: boolean;
}

export function useTaskPolling(
  taskId: string | null,
  intervalMs: number = DEFAULT_INTERVAL,
): PollingState {
  const [state, setState] = useState<PollingState>({
    task: null,
    loading: false,
    error: null,
    isTerminal: false,
  });

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  const poll = useCallback(async () => {
    if (!taskId || !mountedRef.current) return;

    try {
      const { data } = await axios.get<TaskState>(`/api/tasks/${taskId}`);
      if (!mountedRef.current) return;

      const terminal = TERMINAL_STATES.includes(data.status);
      setState({ task: data, loading: false, error: null, isTerminal: terminal });

      if (terminal && intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    } catch (err: unknown) {
      if (!mountedRef.current) return;
      const msg = err instanceof Error ? err.message : 'Polling failed';
      setState(prev => ({ ...prev, error: msg, loading: false }));
    }
  }, [taskId]);

  useEffect(() => {
    mountedRef.current = true;

    if (!taskId) {
      setState({ task: null, loading: false, error: null, isTerminal: false });
      return;
    }

    setState(prev => ({ ...prev, loading: true }));
    void poll();
    intervalRef.current = setInterval(poll, intervalMs);

    return () => {
      mountedRef.current = false;
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, [taskId, intervalMs, poll]);

  return state;
}
