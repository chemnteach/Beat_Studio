import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import App from '../App'

// Mock all axios calls
vi.mock('axios', () => ({
  default: {
    post: vi.fn().mockResolvedValue({ data: {} }),
    get: vi.fn().mockResolvedValue({ data: { songs: [], total: 0, loras: [], styles: [], available: false, name: 'none', total_gb: 0, used_gb: 0, free_gb: 0 } }),
    delete: vi.fn().mockResolvedValue({}),
  },
}))

describe('App', () => {
  it('renders the app header', () => {
    render(<App />)
    expect(screen.getByTestId('app-header')).toBeInTheDocument()
    expect(screen.getByText('Beat Studio')).toBeInTheDocument()
  })

  it('renders tab navigation', () => {
    render(<App />)
    expect(screen.getByTestId('tab-nav')).toBeInTheDocument()
  })

  it('has Upload & Song nav button', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: 'Upload & Song' })).toBeInTheDocument()
  })

  it('has all 5 tab nav buttons', () => {
    render(<App />)
    const nav = screen.getByTestId('tab-nav')
    expect(nav.querySelectorAll('button')).toHaveLength(5)
  })

  it('shows audio upload on default upload tab', () => {
    render(<App />)
    expect(screen.getByTestId('audio-upload')).toBeInTheDocument()
  })

  it('switches to mashup tab on click', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Mashup Workshop' }))
    expect(screen.getByTestId('mashup-workshop')).toBeInTheDocument()
  })

  it('shows no-audio message in video tab without upload', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Video Studio' }))
    expect(screen.getByTestId('no-audio-msg')).toBeInTheDocument()
  })

  it('switches to system tab and shows hardware status', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'System Status' }))
    expect(screen.getByTestId('hardware-status')).toBeInTheDocument()
  })

  it('renders tagline', () => {
    render(<App />)
    expect(screen.getByText(/AI-powered music video production/)).toBeInTheDocument()
  })
})
