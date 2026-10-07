import { describe, expect, it } from 'vitest'
import {
  addRunUsage,
  emptyRunUsage,
  isMeasured,
  totalTokens,
} from './runUsage'

const main = 'run-1'
const sub = 'run-1:researcher'

describe('run accounting', () => {
  it('starts at zero', () => {
    const usage = emptyRunUsage()

    expect(usage.calls).toBe(0)
    expect(totalTokens(usage)).toBe(0)
    expect(isMeasured(usage)).toBe(false)
  })

  it('accumulates across calls of one run', () => {
    let usage = emptyRunUsage()

    usage = addRunUsage(usage, main, {
      promptTokens: 1000,
      completionTokens: 100,
      estimatedCost: 0.005,
    })
    usage = addRunUsage(usage, main, {
      promptTokens: 2000,
      completionTokens: 50,
      estimatedCost: 0.01,
    })

    expect(usage.promptTokens).toBe(3000)
    expect(usage.completionTokens).toBe(150)
    expect(usage.calls).toBe(2)
    expect(usage.estimatedCost).toBeCloseTo(0.015, 6)
  })

  // The distinction the module exists to preserve: a provider that
  // says nothing is not a provider that says zero.
  it('counts a silent provider without inventing tokens', () => {
    let usage = emptyRunUsage()

    usage = addRunUsage(usage, main, {
      promptTokens: null,
      completionTokens: null,
      estimatedCost: null,
    })

    expect(usage.calls).toBe(1)
    expect(usage.callsWithoutUsage).toBe(1)
    expect(usage.promptTokens).toBe(0)
    expect(isMeasured(usage)).toBe(false)
  })

  it('marks a partly measured run as measured', () => {
    let usage = emptyRunUsage()

    usage = addRunUsage(usage, main, {
      promptTokens: null,
      completionTokens: null,
      estimatedCost: null,
    })
    usage = addRunUsage(usage, main, {
      promptTokens: 500,
      completionTokens: 20,
      estimatedCost: 0.002,
    })

    expect(isMeasured(usage)).toBe(true)
    expect(usage.callsWithoutUsage).toBe(1)
  })

  it('leaves the cost unset when no price is configured', () => {
    let usage = emptyRunUsage()

    usage = addRunUsage(usage, main, {
      promptTokens: 1000,
      completionTokens: 100,
      estimatedCost: null,
    })

    // null, not 0: an unpriced run is not a free one.
    expect(usage.estimatedCost).toBeNull()
  })

  // A partial sum understates the bill and still looks like a bill.
  it('drops the whole total when one call has no price', () => {
    let usage = emptyRunUsage()

    usage = addRunUsage(usage, main, {
      promptTokens: 1000,
      completionTokens: 0,
      estimatedCost: 0.003,
    })
    usage = addRunUsage(usage, main, {
      promptTokens: 1000,
      completionTokens: 0,
      estimatedCost: null,
    })

    expect(usage.estimatedCost).toBeNull()
  })

  // Folding subagent work into the parent shows a total no provider
  // ever reported.
  it('keeps a subagent out of the main run total', () => {
    let usage = emptyRunUsage()

    usage = addRunUsage(usage, main, {
      promptTokens: 1000,
      completionTokens: 100,
      estimatedCost: 0.005,
    })
    usage = addRunUsage(usage, sub, {
      promptTokens: 9000,
      completionTokens: 900,
      estimatedCost: 0.05,
    })

    expect(usage.promptTokens).toBe(1000)
    expect(usage.calls).toBe(1)
  })
})
