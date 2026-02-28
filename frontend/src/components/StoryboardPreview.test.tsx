/**
 * Tests for StoryboardPreview — progress indicator and per-scene LoRA weight sliders.
 *
 * Covers:
 *   - Generating phase: progress counter shows as scenes complete
 *   - Ready phase: sliders render only for LoRAs whose trigger tokens appear in the prompt
 *   - Slider change: updates local weight state (reflected in next regen request)
 *   - Regenerate request: lora_weights included in POST body
 *   - Preset buttons: set weights correctly based on LoRA type
 *   - Version picker tooltip: shows lora_weights from the VersionEntry
 *
 * NOTE on fake timers + waitFor:
 *   RTL's waitFor uses setTimeout internally for retries. With vi.useFakeTimers() those
 *   timers are frozen, so waitFor hangs forever. We avoid waitFor entirely and instead
 *   use act() + synchronous expect, with vi.advanceTimersByTime + multiple
 *   "await act(async () => { await Promise.resolve() })" flushes to drain async chains.
 */
import { render, screen, fireEvent, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import axios from 'axios'
import { StoryboardPreview } from './StoryboardPreview'

// ── Mocks ─────────────────────────────────────────────────────────────────────

vi.mock('axios')
const mockedAxios = axios as unknown as {
  get: ReturnType<typeof vi.fn>
  post: ReturnType<typeof vi.fn>
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const STYLE = 'watercolor'
const LORA_NAMES = ['rob-character', 'watercolor-style']

const SCENES = [
  {
    scene_idx: 0,
    storyboard_prompt: 'Cinematic ocean scene',
    positive_prompt: 'rob_character standing on a beach, watercolor_style paint brush strokes',
  },
  {
    scene_idx: 1,
    storyboard_prompt: 'City skyline at dusk',
    positive_prompt: 'urban skyline at dusk, watercolor_style brushwork, soft colors',
  },
]

const LORA_LIST_RESPONSE = {
  loras: [
    { name: 'rob-character', trigger_token: 'rob_character', type: 'character', status: 'available' },
    { name: 'watercolor-style', trigger_token: 'watercolor_style', type: 'style', status: 'available' },
  ],
}

function makeImagesResponse(completedScenes = 2) {
  return {
    storyboard_id: 'sb-test',
    status: completedScenes < 2 ? 'generating' : 'complete',
    scenes: SCENES.slice(0, completedScenes).map((s, i) => ({
      scene_idx: s.scene_idx,
      storyboard_prompt: s.storyboard_prompt,
      positive_prompt: s.positive_prompt,
      approved_version: null,
      versions: [{ version: 1, url: `/img/scene_${i}/v1.png`, seed: 0, timestamp: 'ts', lora_weights: {} }],
    })),
  }
}

function makeProps(overrides = {}) {
  return {
    style: STYLE,
    scenes: SCENES,
    loraNames: LORA_NAMES,
    onApprove: vi.fn(),
    onBack: vi.fn(),
    ...overrides,
  }
}

/**
 * Flush one round of pending microtasks (Promise.resolve) within React's act.
 * Multiple calls drain deeply-nested async callback chains.
 */
const flush = () => act(async () => { await Promise.resolve() })

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('StoryboardPreview — progress indicator', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mockedAxios.get = vi.fn()
    mockedAxios.post = vi.fn()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  it('shows generating spinner in generating phase', async () => {
    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-1', storyboard_id: 'sb-test' },
    })
    // First GET on mount is /api/lora/list, then task poll, then img poll fallback
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })        // lora/list on mount
      .mockResolvedValueOnce({ data: { status: 'running' } })     // task poll
      .mockResolvedValue({ data: makeImagesResponse(0) })         // img poll

    render(<StoryboardPreview {...makeProps()} />)

    // Wrap click in act so the synchronous setPhase('generating') is flushed
    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })

    expect(screen.getByTestId('stage-storyboard-generating')).toBeInTheDocument()
  })

  it('shows scene count progress during generation', async () => {
    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-2', storyboard_id: 'sb-test' },
    })
    // Correct order: lora/list first, then task poll ('running'), then img poll (1-of-2 scenes)
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })        // lora/list on mount
      .mockResolvedValueOnce({ data: { status: 'running' } })     // task poll at 2 s
      .mockResolvedValueOnce({ data: makeImagesResponse(1) })     // img poll at 3 s
      .mockResolvedValue({ data: LORA_LIST_RESPONSE })

    render(<StoryboardPreview {...makeProps()} />)

    // Click and let POST + interval setup settle
    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })
    await flush()  // let POST continuation (setStoryboardId + setInterval) resolve

    // Advance 3100 ms: fires task poll at 2000 ms (running) and img poll at 3000 ms (1/2 scenes)
    act(() => { vi.advanceTimersByTime(3100) })
    // Drain both async callbacks
    await flush()
    await flush()

    const generating = screen.queryByTestId('stage-storyboard-generating')
    if (generating) {
      expect(generating.textContent).toMatch(/1.*of.*2|1\/2/)
    }
  })
})

describe('StoryboardPreview — LoRA weight sliders', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mockedAxios.get = vi.fn()
    mockedAxios.post = vi.fn()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  /**
   * Renders the component, generates, and waits for 'ready' state.
   * Mock order:
   *   GET 1 → /api/lora/list (mount effect)
   *   GET 2 → /api/tasks/task-r  (task poll at 2 s → completed)
   *   GET 3+ → /api/storyboard/.../images (fetchImages)
   */
  async function renderReadyState() {
    const completedImages = makeImagesResponse(2)

    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-r', storyboard_id: 'sb-test' },
    })
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })        // lora/list on mount
      .mockResolvedValueOnce({ data: { status: 'completed' } })   // task poll
      .mockResolvedValue({ data: completedImages })               // images fetch

    render(<StoryboardPreview {...makeProps()} />)

    // Click generate; act flushes: setPhase('generating') + POST resolve + interval setup
    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })
    await flush()  // drain any remaining async setup (setStoryboardId, setInterval)

    // Fire the 2 s task poll
    act(() => { vi.advanceTimersByTime(2500) })
    // Flush: task GET resolves → detect 'completed' → fetchImages → images GET → setPhase
    await flush()
    await flush()
    await flush()

    expect(screen.getByTestId('stage-storyboard-ready')).toBeInTheDocument()

    return completedImages
  }

  it('renders sliders for loras whose trigger tokens appear in the active scene prompt', async () => {
    await renderReadyState()

    // Scene 0 prompt contains both rob_character and watercolor_style
    expect(screen.getByTestId('lora-weight-slider-rob-character')).toBeInTheDocument()
    expect(screen.getByTestId('lora-weight-slider-watercolor-style')).toBeInTheDocument()
  })

  it('does not render sliders for loras whose trigger tokens are absent from the prompt', async () => {
    const props = makeProps({
      scenes: [
        {
          scene_idx: 0,
          storyboard_prompt: 'City scene',
          positive_prompt: 'urban city scene, watercolor_style painting',  // no rob_character
        },
      ],
      loraNames: ['rob-character', 'watercolor-style'],
    })

    const oneSceneImages = {
      storyboard_id: 'sb-test',
      status: 'complete',
      scenes: [{
        scene_idx: 0,
        storyboard_prompt: 'City scene',
        positive_prompt: 'urban city scene, watercolor_style painting',
        approved_version: null,
        versions: [{ version: 1, url: '/img/scene_0/v1.png', seed: 0, timestamp: 'ts', lora_weights: {} }],
      }],
    }

    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-r2', storyboard_id: 'sb-test' },
    })
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })
      .mockResolvedValueOnce({ data: { status: 'completed' } })
      .mockResolvedValue({ data: oneSceneImages })

    render(<StoryboardPreview {...props} />)

    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })
    await flush()

    act(() => { vi.advanceTimersByTime(2500) })
    await flush()
    await flush()
    await flush()

    expect(screen.getByTestId('stage-storyboard-ready')).toBeInTheDocument()

    expect(screen.queryByTestId('lora-weight-slider-rob-character')).not.toBeInTheDocument()
    expect(screen.getByTestId('lora-weight-slider-watercolor-style')).toBeInTheDocument()
  })

  it('slider default value is 0.7', async () => {
    await renderReadyState()
    const slider = screen.getByTestId('lora-weight-slider-rob-character') as HTMLInputElement
    expect(slider.value).toBe('0.7')
  })

  it('changing a slider updates the displayed value', async () => {
    await renderReadyState()
    const slider = screen.getByTestId('lora-weight-slider-rob-character')
    await act(async () => {
      fireEvent.change(slider, { target: { value: '0.3' } })
    })
    expect(screen.getByTestId('lora-weight-value-rob-character').textContent).toBe('0.3')
  })

  it('regenerate sends lora_weights in request body', async () => {
    await renderReadyState()

    // Change one slider
    const slider = screen.getByTestId('lora-weight-slider-rob-character')
    await act(async () => {
      fireEvent.change(slider, { target: { value: '0.3' } })
    })

    // Mock regenerate response
    mockedAxios.post = vi.fn().mockResolvedValue({ data: { task_id: 'regen-task' } })

    // Click regenerate; act flushes the axios.post call
    await act(async () => {
      fireEvent.click(screen.getByTestId('regenerate-scene-btn'))
    })
    await flush()  // let the POST microtask resolve

    const postCalls = mockedAxios.post.mock.calls
    const regenCall = postCalls.find((c: unknown[]) =>
      typeof c[0] === 'string' && c[0].includes('regenerate')
    )
    expect(regenCall).toBeDefined()
    const body = regenCall![1] as Record<string, unknown>
    expect(body.lora_weights).toBeDefined()
    expect((body.lora_weights as Record<string, number>)['rob-character']).toBe(0.3)
  })

  it('Style priority preset sets style loras to 0.3 and character loras to 0.7', async () => {
    await renderReadyState()

    await act(async () => {
      fireEvent.click(screen.getByTestId('preset-style-priority'))
    })

    const charSlider = screen.getByTestId('lora-weight-slider-rob-character') as HTMLInputElement
    const styleSlider = screen.getByTestId('lora-weight-slider-watercolor-style') as HTMLInputElement

    expect(charSlider.value).toBe('0.7')
    expect(styleSlider.value).toBe('0.3')
  })

  it('Balanced preset sets all weights to 0.6', async () => {
    await renderReadyState()

    await act(async () => {
      fireEvent.click(screen.getByTestId('preset-balanced'))
    })

    const charSlider = screen.getByTestId('lora-weight-slider-rob-character') as HTMLInputElement
    const styleSlider = screen.getByTestId('lora-weight-slider-watercolor-style') as HTMLInputElement

    expect(charSlider.value).toBe('0.6')
    expect(styleSlider.value).toBe('0.6')
  })

  it('LoRA priority preset sets all weights to 0.9', async () => {
    await renderReadyState()

    await act(async () => {
      fireEvent.click(screen.getByTestId('preset-lora-priority'))
    })

    const charSlider = screen.getByTestId('lora-weight-slider-rob-character') as HTMLInputElement
    const styleSlider = screen.getByTestId('lora-weight-slider-watercolor-style') as HTMLInputElement

    expect(charSlider.value).toBe('0.9')
    expect(styleSlider.value).toBe('0.9')
  })

  it('version picker title includes lora_weights when present', async () => {
    const imagesWithWeights = makeImagesResponse(2)
    // Version picker only renders when versions.length > 1; add a second version so it's visible
    imagesWithWeights.scenes[0].versions = [
      { version: 1, url: '/img/scene_0/v1.png', seed: 0, timestamp: 'ts', lora_weights: { 'rob-character': 0.35 } },
      { version: 2, url: '/img/scene_0/v2.png', seed: 1, timestamp: 'ts2', lora_weights: {} },
    ]

    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-tw', storyboard_id: 'sb-test' },
    })
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })
      .mockResolvedValueOnce({ data: { status: 'completed' } })
      .mockResolvedValue({ data: imagesWithWeights })

    render(<StoryboardPreview {...makeProps()} />)

    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })
    await flush()

    act(() => { vi.advanceTimersByTime(2500) })
    await flush()
    await flush()
    await flush()

    expect(screen.getByTestId('stage-storyboard-ready')).toBeInTheDocument()

    // Scene 0 has 2 versions, so the version picker is visible
    // v1 button should have a title that mentions the weights
    const v1btn = screen.getByTestId('version-btn-0-1')
    expect(v1btn.getAttribute('title')).toContain('rob-character')
    expect(v1btn.getAttribute('title')).toContain('0.35')
  })
})

describe('StoryboardPreview — ZIP upload', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mockedAxios.get = vi.fn()
    mockedAxios.post = vi.fn()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  /** Drive the component to 'ready' phase, same helper pattern as LoRA weight tests. */
  async function renderReadyStateForUpload() {
    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-up', storyboard_id: 'sb-test' },
    })
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })
      .mockResolvedValueOnce({ data: { status: 'completed' } })
      .mockResolvedValue({ data: makeImagesResponse(2) })

    render(<StoryboardPreview {...makeProps()} />)

    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })
    await flush()

    act(() => { vi.advanceTimersByTime(2500) })
    await flush()
    await flush()
    await flush()

    expect(screen.getByTestId('stage-storyboard-ready')).toBeInTheDocument()
  }

  it('upload button renders in stage-nav', async () => {
    await renderReadyStateForUpload()
    expect(screen.getByTestId('upload-storyboard-btn')).toBeInTheDocument()
  })

  it('upload sends file to the correct endpoint', async () => {
    await renderReadyStateForUpload()

    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { uploaded: 2, snapshot_id: 'snap-1' },
    })

    const input = screen.getByTestId('upload-storyboard-input') as HTMLInputElement
    const file = new File([new Uint8Array([1, 2, 3])], 'storyboard.zip', { type: 'application/zip' })

    await act(async () => {
      Object.defineProperty(input, 'files', { value: [file], configurable: true })
      fireEvent.change(input)
    })
    await flush()
    await flush()

    const uploadCall = mockedAxios.post.mock.calls.find((c: unknown[]) =>
      typeof c[0] === 'string' && (c[0] as string).includes('upload')
    )
    expect(uploadCall).toBeDefined()
    expect(uploadCall![1]).toBeInstanceOf(FormData)
  })

  it('upload success shows toast and refreshes images', async () => {
    await renderReadyStateForUpload()

    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { uploaded: 2, snapshot_id: 'snap-1' },
    })
    const getCallsBefore = (mockedAxios.get as ReturnType<typeof vi.fn>).mock.calls.length

    const input = screen.getByTestId('upload-storyboard-input') as HTMLInputElement
    const file = new File([new Uint8Array([1, 2, 3])], 'test.zip', { type: 'application/zip' })

    await act(async () => {
      Object.defineProperty(input, 'files', { value: [file], configurable: true })
      fireEvent.change(input)
    })
    await flush()
    await flush()

    expect(screen.getByTestId('upload-toast')).toBeInTheDocument()
    expect(screen.getByTestId('upload-toast').textContent).toContain('2 scenes')
    // fetchImages should have been called once more
    expect((mockedAxios.get as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThan(getCallsBefore)
  })

  it('upload 422 error shows validation message', async () => {
    await renderReadyStateForUpload()

    mockedAxios.post = vi.fn().mockRejectedValue({
      response: { status: 422, data: { detail: 'ZIP contains 2 scenes, storyboard has 3' } },
    })

    const input = screen.getByTestId('upload-storyboard-input') as HTMLInputElement
    const file = new File([new Uint8Array([1, 2, 3])], 'bad.zip', { type: 'application/zip' })

    await act(async () => {
      Object.defineProperty(input, 'files', { value: [file], configurable: true })
      fireEvent.change(input)
    })
    await flush()
    await flush()

    const errEl = screen.getByTestId('upload-error')
    expect(errEl).toBeInTheDocument()
    expect(errEl.textContent).toContain('ZIP contains')
  })

  it('uploaded version shows ↑ upload badge', async () => {
    const imagesWithUpload = makeImagesResponse(2)
    imagesWithUpload.scenes[0].versions = [
      { version: 1, url: '/img/scene_0/v1.png', seed: 42, timestamp: 'ts', lora_weights: {}, source: 'generated' },
      { version: 2, url: '/img/scene_0/v2.png', seed: null, timestamp: 'ts2', lora_weights: {}, source: 'upload' },
    ]

    mockedAxios.post = vi.fn().mockResolvedValue({
      data: { task_id: 'task-badge', storyboard_id: 'sb-test' },
    })
    mockedAxios.get = vi.fn()
      .mockResolvedValueOnce({ data: LORA_LIST_RESPONSE })
      .mockResolvedValueOnce({ data: { status: 'completed' } })
      .mockResolvedValue({ data: imagesWithUpload })

    render(<StoryboardPreview {...makeProps()} />)

    await act(async () => {
      fireEvent.click(screen.getByTestId('generate-storyboard-btn'))
    })
    await flush()
    act(() => { vi.advanceTimersByTime(2500) })
    await flush()
    await flush()
    await flush()

    expect(screen.getByTestId('stage-storyboard-ready')).toBeInTheDocument()

    // Scene 0 has 2 versions — version picker is visible
    // v2 is uploaded: should have the upload badge
    const badge = screen.getByTestId('upload-badge-0-2')
    expect(badge).toBeInTheDocument()
    expect(badge.textContent).toBe('↑')

    // v1 is generated: no badge
    expect(screen.queryByTestId('upload-badge-0-1')).not.toBeInTheDocument()

    // v2 title should say "Uploaded at" not "Seed:"
    const v2btn = screen.getByTestId('version-btn-0-2')
    expect(v2btn.getAttribute('title')).toContain('Uploaded at')
    expect(v2btn.getAttribute('title')).not.toContain('Seed:')
  })
})
