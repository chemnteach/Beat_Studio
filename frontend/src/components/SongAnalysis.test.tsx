import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { SongAnalysis } from './SongAnalysis'
import type { SongAnalysis as SongAnalysisType } from '../types'

const mockAnalysis: SongAnalysisType = {
  audio_id: 'test-001',
  title: 'Test Song',
  artist: 'Test Artist',
  duration: 180,
  bpm: 128,
  key: 'G major',
  camelot: '9B',
  mood_summary: 'upbeat energetic',
  energy: 0.8,
  sections: [
    { start: 0, end: 30, section_type: 'verse', energy_level: 0.5 },
    { start: 30, end: 60, section_type: 'chorus', energy_level: 0.9 },
  ],
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
