import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { VideoPreview } from './VideoPreview'

describe('VideoPreview', () => {
  it('renders download options', () => {
    render(<VideoPreview videoId="vid-001" />)
    expect(screen.getByTestId('download-options')).toBeInTheDocument()
  })

  it('has download links for default platforms', () => {
    render(<VideoPreview videoId="vid-001" />)
    expect(screen.getByTestId('download-youtube')).toBeInTheDocument()
    expect(screen.getByTestId('download-tiktok')).toBeInTheDocument()
  })

  it('renders video player when url provided', () => {
    render(<VideoPreview videoId="vid-001" videoUrl="/path/to/video.mp4" />)
    expect(screen.getByTestId('video-player')).toBeInTheDocument()
  })

  it('renders placeholder when no url', () => {
    render(<VideoPreview videoId="vid-001" />)
    expect(screen.getByTestId('video-placeholder')).toBeInTheDocument()
  })

  it('download links include correct video id', () => {
    render(<VideoPreview videoId="vid-123" />)
    const youtubeLink = screen.getByTestId('download-youtube')
    expect(youtubeLink).toHaveAttribute('href', expect.stringContaining('vid-123'))
  })
})
