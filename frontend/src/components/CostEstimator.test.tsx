import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { CostEstimator } from './CostEstimator'
import type { ExecutionPlan } from '../types'

const plan: ExecutionPlan = {
  plan_id: 'p1',
  backend: 'animatediff',
  num_scenes: 24,
  estimated_time_sec: 1200,
  estimated_cost_usd: 0,
  is_local: true,
  scenes: [],
}

describe('CostEstimator', () => {
  it('shows empty state when no plan', () => {
    render(<CostEstimator plan={null} />)
    expect(screen.getByTestId('cost-estimator')).toHaveClass('empty')
  })

  it('shows cost table when plan provided', () => {
    render(<CostEstimator plan={plan} />)
    expect(screen.getByTestId('cost-table')).toBeInTheDocument()
  })

  it('shows Free for local plan', () => {
    render(<CostEstimator plan={plan} />)
    expect(screen.getByTestId('est-cost')).toHaveTextContent('Free')
  })

  it('shows backend name', () => {
    render(<CostEstimator plan={plan} />)
    expect(screen.getByTestId('est-backend')).toHaveTextContent('animatediff')
  })

  it('formats time correctly for long estimate', () => {
    render(<CostEstimator plan={plan} />)
    expect(screen.getByTestId('est-time')).toHaveTextContent('20m')
  })
})
