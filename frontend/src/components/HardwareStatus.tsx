/**
 * HardwareStatus — displays GPU/VRAM status.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { GPUStatus } from '../types';

export function HardwareStatus() {
  const [gpu, setGpu] = useState<GPUStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void fetchGPU();
    const interval = setInterval(() => void fetchGPU(), 10_000);
    return () => clearInterval(interval);
  }, []);

  const fetchGPU = async () => {
    try {
      const { data } = await axios.get<GPUStatus>('/api/system/gpu');
      setGpu(data);
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="hardware-status loading" data-testid="hardware-status">Loading GPU info…</div>;

  if (!gpu?.available) {
    return (
      <div className="hardware-status no-gpu" data-testid="hardware-status">
        No GPU detected. Cloud backends will be used.
      </div>
    );
  }

  const usedPct = Math.round((gpu.used_gb / gpu.total_gb) * 100);

  return (
    <div className="hardware-status" data-testid="hardware-status">
      <h4>GPU Status</h4>
      <div className="gpu-info">
        <span className="gpu-name" data-testid="gpu-name">{gpu.name}</span>
        <div className="vram-bar-container" data-testid="vram-bar-container">
          <div
            className="vram-bar"
            style={{ width: `${usedPct}%`, backgroundColor: usedPct > 80 ? '#dc3545' : '#198754' }}
            data-testid="vram-bar"
          />
        </div>
        <span className="vram-text" data-testid="vram-text">
          {gpu.used_gb.toFixed(1)} / {gpu.total_gb.toFixed(1)} GB ({usedPct}%)
        </span>
        <span className="vram-free" data-testid="vram-free">
          {gpu.free_gb.toFixed(1)} GB free
        </span>
      </div>
    </div>
  );
}
