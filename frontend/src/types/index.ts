/**
 * Beat Studio — shared TypeScript types.
 * Mirrors backend Pydantic models for type safety across the API boundary.
 */

// ── Audio & Analysis ──────────────────────────────────────────────────────────

export interface SectionInfo {
  start: number;
  end: number;
  section_type: string;
  energy_level: number;
}

export interface WordTiming {
  word: string;
  start: number;
  end: number;
}

export interface SongAnalysis {
  audio_id: string;
  title: string;
  artist: string;
  duration: number;
  bpm: number;
  key: string;
  camelot: string;
  mood_summary: string;
  energy: number;
  sections: SectionInfo[];
  lyrics: string;
  word_timings: WordTiming[];
}

// ── Mashup ────────────────────────────────────────────────────────────────────

export type MashupType =
  | 'classic'
  | 'stem_swap'
  | 'energy_match'
  | 'adaptive_harmony'
  | 'theme_fusion'
  | 'semantic_aligned'
  | 'role_aware'
  | 'conversational';

export interface SongEntry {
  id: string;
  title: string;
  artist: string;
  bpm: number;
  key: string;
  mood_summary: string;
  duration: number;
}

export interface MatchResult {
  song: SongEntry;
  score: number;
  bpm_match: number;
  key_match: number;
  mood_match: number;
  recommended_type: MashupType;
}

// ── Video ─────────────────────────────────────────────────────────────────────

export interface AnimationStyle {
  name: string;
  display_name: string;
  category: string;
  prompt_suffix: string;
  recommended_backend: string;
  guidance_scale: number;
  best_for: string;
}

export interface ScenePlan {
  scene_index: number;
  backend_name: string;
  estimated_time_sec: number;
  estimated_cost_usd: number;
  resolution: [number, number];
  prompt?: string;
  notes?: string;
}

export interface ExecutionPlan {
  plan_id: string;
  backend: string;
  num_scenes: number;
  estimated_time_sec: number;
  estimated_cost_usd: number;
  is_local: boolean;
  scenes: ScenePlan[];
}

export interface VideoBackendStatus {
  name: string;
  available: boolean;
  vram_required_gb: number;
  cost_per_scene: number;
}

// ── LoRA ──────────────────────────────────────────────────────────────────────

export type LoRAType = 'character' | 'scene' | 'style' | 'identity';

export interface LoRAEntry {
  name: string;
  type: LoRAType;
  trigger_token: string;
  file_path: string;
  weight: number;
  status: 'available' | 'missing' | 'downloading';
  description: string;
  tags: string[];
  source: string;
  source_url?: string;
}

export interface LoRASearchResult {
  name: string;
  source: string;
  url: string;
  type: string;
  description: string;
  trigger_token?: string;
  confidence: number;
}

// ── Nova Fade ─────────────────────────────────────────────────────────────────

export type NovaFadeTheme = 'sponsor_neon' | 'award_elegant' | 'mashup_chaos' | 'chill_lofi';

export interface NovaFadeStatus {
  identity_lora: 'available' | 'missing';
  style_lora: 'available' | 'missing';
  canonical_images: number;
  last_drift_test: string | null;
  constitution_version: string;
}

export interface DriftScorecard {
  s_id: number;
  s_face: number;
  s_sil: number;
  v_batch: number;
  passed: boolean;
  summary: string;
}

// ── Tasks ─────────────────────────────────────────────────────────────────────

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TaskState {
  task_id: string;
  status: TaskStatus;
  progress: number;        // 0–100
  stage?: string;          // current stage label
  result?: unknown;
  error?: string;
  created_at?: string;
}

export interface ProgressEvent {
  task_id: string;
  progress: number;
  stage: string;
  message?: string;
}

// ── System ────────────────────────────────────────────────────────────────────

export interface GPUStatus {
  available: boolean;
  name: string;
  total_gb: number;
  used_gb: number;
  free_gb: number;
}

export interface ModelEntry {
  name: string;
  purpose: string;
  installed: boolean;
  size_gb: number;
  path?: string;
}
