import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ExecutionPlanner } from './ExecutionPlanner'
import type { ExecutionPlan } from '../types'

const localPlan: ExecutionPlan = {
  plan_id: 'plan-001',
  backend: 'animatediff',
  num_scenes: 12,
  estimated_time_sec: 600,
  estimated_cost_usd: 0.0,
  is_local: true,
  scenes: [],
}

const cloudPlan: ExecutionPlan = {
  ...localPlan,
  plan_id: 'plan-002',
  backend: 'wan26_cloud',
  estimated_cost_usd: 1.20,
  is_local: false,
}

describe('ExecutionPlanner', () => {
  it('shows backend name', () => {
    render(<ExecutionPlanner plan={localPlan} onConfirm={() => {}} />)
    expect(screen.getByTestId('plan-backend')).toHaveTextContent('animatediff')
  })

  it('shows scene count', () => {
    render(<ExecutionPlanner plan={localPlan} onConfirm={() => {}} />)
    expect(screen.getByTestId('plan-scenes')).toHaveTextContent('12')
  })

  it('shows Free for local plan', () => {
    render(<ExecutionPlanner plan={localPlan} onConfirm={() => {}} />)
    expect(screen.getByTestId('plan-cost')).toHaveTextContent('Free')
  })

  it('shows cost for cloud plan', () => {
    render(<ExecutionPlanner plan={cloudPlan} onConfirm={() => {}} />)
    expect(screen.getByTestId('plan-cost')).toHaveTextContent('$1.20')
  })

  it('shows cloud warning for cloud plan', () => {
    render(<ExecutionPlanner plan={cloudPlan} onConfirm={() => {}} />)
    expect(screen.getByTestId('cloud-warning')).toBeInTheDocument()
  })

  it('does not show cloud warning for local plan', () => {
    render(<ExecutionPlanner plan={localPlan} onConfirm={() => {}} />)
    expect(screen.queryByTestId('cloud-warning')).not.toBeInTheDocument()
  })

  it('calls onConfirm with plan_id when confirmed', () => {
    const onConfirm = vi.fn()
    render(<ExecutionPlanner plan={localPlan} onConfirm={onConfirm} />)
    fireEvent.click(screen.getByTestId('confirm-plan-btn'))
    expect(onConfirm).toHaveBeenCalledWith('plan-001')
  })
})
