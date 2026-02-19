import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { AudioUpload } from './AudioUpload'

// Mock axios
vi.mock('axios', () => ({
  default: {
    post: vi.fn().mockResolvedValue({ data: { audio_id: 'test-id', filename: 'test.mp3' } }),
    get: vi.fn(),
  },
}))

describe('AudioUpload', () => {
  it('renders the drop zone', () => {
    render(<AudioUpload onUploadComplete={() => {}} />)
    expect(screen.getByTestId('audio-upload')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /drop audio/i })).toBeInTheDocument()
  })

  it('renders the YouTube URL input', () => {
    render(<AudioUpload onUploadComplete={() => {}} />)
    expect(screen.getByTestId('youtube-url-input')).toBeInTheDocument()
  })

  it('enables ingest button when URL is entered', () => {
    render(<AudioUpload onUploadComplete={() => {}} />)
    const input = screen.getByTestId('youtube-url-input')
    fireEvent.change(input, { target: { value: 'https://youtube.com/watch?v=abc' } })
    const ingestBtn = screen.getByText('Ingest')
    expect(ingestBtn).not.toBeDisabled()
  })

  it('has a hidden file input for accepted formats', () => {
    render(<AudioUpload onUploadComplete={() => {}} />)
    const fileInput = screen.getByTestId('file-input')
    expect(fileInput).toHaveAttribute('accept', '.mp3,.wav,.flac,.m4a')
  })
})
