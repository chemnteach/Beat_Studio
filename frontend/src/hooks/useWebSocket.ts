/**
 * useWebSocket — real-time task progress via WebSocket.
 *
 * Connects to /ws/{taskId} and streams ProgressEvent updates.
 * Automatically reconnects on disconnect (up to maxRetries times).
 * Falls back gracefully when WebSocket is unavailable.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { ProgressEvent, TaskStatus } from '../types';

const WS_BASE = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000';
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;

export interface WebSocketState {
  progress: number;
  stage: string;
  status: TaskStatus;
  lastEvent: ProgressEvent | null;
  connected: boolean;
  error: string | null;
}

export function useWebSocket(taskId: string | null): WebSocketState {
  const [state, setState] = useState<WebSocketState>({
    progress: 0,
    stage: '',
    status: 'pending',
    lastEvent: null,
    connected: false,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!taskId || !mountedRef.current) return;

    const url = `${WS_BASE}/ws/${taskId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      retriesRef.current = 0;
      if (mountedRef.current) {
        setState(prev => ({ ...prev, connected: true, error: null }));
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(event.data as string) as ProgressEvent;
        setState(prev => ({
          ...prev,
          progress: data.progress,
          stage: data.stage,
          lastEvent: data,
        }));
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setState(prev => ({ ...prev, connected: false }));
      if (retriesRef.current < MAX_RETRIES) {
        retriesRef.current += 1;
        setTimeout(connect, RETRY_DELAY_MS);
      }
    };

    ws.onerror = () => {
      if (mountedRef.current) {
        setState(prev => ({
          ...prev,
          connected: false,
          error: 'WebSocket connection failed',
        }));
      }
    };
  }, [taskId]);

  useEffect(() => {
    mountedRef.current = true;
    if (taskId) {
      connect();
    }
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [taskId, connect]);

  return state;
}
