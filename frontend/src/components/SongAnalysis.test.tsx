import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { SongAnalysis } from './SongAnalysis'
import type { SongAnalysis as SongAnalysisType } from '../types'

const mockSection = (type: string, start: number, end: number) => ({
  section_type: type, start_sec: start, end_sec: end,
  duration_sec: end - start, energy_level: 0.5, spectral_centroid: 0,
  tempo_stability: 0, vocal_density: 'medium', vocal_intensity: 0,
  lyrical_content: '', emotional_tone: 'neutral', lyrical_function: 'narrative', themes: [],
});

const mockAnalysis: SongAnalysisType = {
  audio_id: 'test-001',
  title: 'Test Song',
  artist: 'Test Artist',
  duration_sec: 180,
  bpm: 128,
  key: 'G major',
  camelot: '9B',
  mood_summary: 'upbeat energetic',
  energy_level: 0.8,
  sections: [mockSection('verse', 0, 30), mockSection('chorus', 30, 60)],
  lyrics: 'Sample lyrics here',
  word_timings: [],
}

describe('SongAnalysis', () => {
  it('renders the song title and artist', () => {
    render(
      <SongAnalysis
        analysis={mockAnalysis}
        onCreateVideo={() => {}}
        onCreateMashup={() => {}}
      />
    )
    expect(screen.getByText(/Test Song/)).toBeInTheDocument()
    expect(screen.getByText(/Test Artist/)).toBeInTheDocument()
  })

  it('displays BPM', () => {
    render(<SongAnalysis analysis={mockAnalysis} onCreateVideo={() => {}} onCreateMashup={() => {}} />)
    expect(screen.getByTestId('analysis-metrics')).toHaveTextContent('128')
  })

  it('shows section timeline', () => {
    render(<SongAnalysis analysis={mockAnalysis} onCreateVideo={() => {}} onCreateMashup={() => {}} />)
    expect(screen.getByTestId('section-timeline')).toBeInTheDocument()
    expect(screen.getByTestId('section-0')).toBeInTheDocument()
    expect(screen.getByTestId('section-1')).toBeInTheDocument()
  })

  it('shows lyrics preview', () => {
    render(<SongAnalysis analysis={mockAnalysis} onCreateVideo={() => {}} onCreateMashup={() => {}} />)
    expect(screen.getByTestId('lyrics-preview')).toBeInTheDocument()
  })

  it('calls onCreateVideo when button clicked', () => {
    const onCreateVideo = vi.fn()
    render(<SongAnalysis analysis={mockAnalysis} onCreateVideo={onCreateVideo} onCreateMashup={() => {}} />)
    fireEvent.click(screen.getByTestId('create-video-btn'))
    expect(onCreateVideo).toHaveBeenCalledOnce()
  })

  it('calls onCreateMashup when button clicked', () => {
    const onCreateMashup = vi.fn()
    render(<SongAnalysis analysis={mockAnalysis} onCreateVideo={() => {}} onCreateMashup={onCreateMashup} />)
    fireEvent.click(screen.getByTestId('create-mashup-btn'))
    expect(onCreateMashup).toHaveBeenCalledOnce()
  })
})
