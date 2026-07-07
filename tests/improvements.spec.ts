import { describe, expect, it } from 'vitest'
import type { StatBucket } from '../src/shared/types'
import { bucketStd, summarizeTrajectory } from '../src/stats/chartSeries'
import { annealMultiplier } from '../src/agents/ntuple/anneal'
import { defaultNTupleConfig, nextPresetUp } from '../src/agents/factory'
import { NTupleAgent } from '../src/agents/ntuple/tdLearner'

function bucket(partial: Partial<StatBucket>): StatBucket {
  return {
    bucket: 0,
    n: 100,
    scoreSum: 0,
    scoreSqSum: 0,
    scoreMin: 0,
    scoreMax: 0,
    movesSum: 0,
    maxTileHist: new Array(16).fill(0),
    tdErrAvg: 0,
    epsAvg: 0,
    lrAvg: 0,
    wallMs: 0,
    endTs: 0,
    ...partial,
  }
}

describe('bucketStd', () => {
  it('computes the standard deviation from stored sum-of-squares', () => {
    // scores [10, 20, 30] → mean 20, variance 200/3, std ≈ 8.165
    const b = bucket({ n: 3, scoreSum: 60, scoreSqSum: 1400 })
    expect(bucketStd(b)).toBeCloseTo(Math.sqrt(200 / 3), 4)
  })
  it('is 0 for empty or degenerate buckets', () => {
    expect(bucketStd(bucket({ n: 0 }))).toBe(0)
    expect(bucketStd(bucket({ n: 5, scoreSum: 500, scoreSqSum: 50000 }))).toBe(0) // all equal 100
  })
})

describe('summarizeTrajectory — relative flat threshold', () => {
  const xs = Array.from({ length: 10 }, (_, i) => i * 100 + 100)

  it('labels a big model gaining +400/1k as converged, not rising', () => {
    // avg ~35k, slope 0.4/game (+400/1k). Old fixed 250 threshold said "rising";
    // relative threshold (2% of 35k = 700) correctly says converged.
    const ys = xs.map((x) => 35_000 + 0.4 * x)
    const s = summarizeTrajectory(xs, ys, 10)!
    expect(Math.round(s.deltaPerThousandGames)).toBe(400)
    expect(s.label).toBe('converged')
  })

  it('labels a small model gaining the same +400/1k as still improving (not converged)', () => {
    // Same absolute slope as the big model above, but at a 1.5k average the
    // relative threshold (max(250, 2%·1500=30)=250) puts it above flat.
    const ys = xs.map((x) => 1500 + 0.4 * x)
    const s = summarizeTrajectory(xs, ys, 10)!
    expect(s.label).toBe('converging')
    expect(s.label).not.toBe('converged')
  })

  it('labels a clearly declining curve as falling', () => {
    const ys = xs.map((x) => 40_000 - 2 * x) // −2000/1k, well past −rel
    expect(summarizeTrajectory(xs, ys, 10)!.label).toBe('falling')
  })
})

describe('annealMultiplier', () => {
  const anneal = { halfLifeGames: 1000, startGame: 5000, floor: 0.1 }
  it('is 1 with no schedule, regardless of games', () => {
    expect(annealMultiplier(999_999, undefined)).toBe(1)
    expect(annealMultiplier(999_999, null)).toBe(1)
  })
  it('is 1 at the start and halves each half-life', () => {
    expect(annealMultiplier(5000, anneal)).toBeCloseTo(1, 6)
    expect(annealMultiplier(6000, anneal)).toBeCloseTo(0.5, 6)
    expect(annealMultiplier(7000, anneal)).toBeCloseTo(0.25, 6)
  })
  it('clamps to the floor and never exceeds 1', () => {
    expect(annealMultiplier(50_000, anneal)).toBe(0.1)
    expect(annealMultiplier(0, anneal)).toBe(1) // before startGame
  })
})

describe('nextPresetUp', () => {
  it('walks the preset ladder and stops at expert', () => {
    expect(nextPresetUp('starter')).toBe('balanced')
    expect(nextPresetUp('balanced')).toBe('expert')
    expect(nextPresetUp('expert')).toBeNull()
  })
})

describe('NTupleAgent annealing wiring', () => {
  it('constant α when annealing is off, no matter how many games', () => {
    const agent = new NTupleAgent(defaultNTupleConfig('starter'))
    const base = agent.getDiagnostics().learningRate
    agent.refreshSchedule(1_000_000)
    expect(agent.getDiagnostics().learningRate).toBe(base)
  })

  it('effective α decays after enabling annealing, and survives serialize/restore', () => {
    const agent = new NTupleAgent(defaultNTupleConfig('starter'))
    const alpha = agent.getDiagnostics().learningRate
    agent.setAnneal({ halfLifeGames: 100, startGame: 0, floor: 0.1 }, 100) // one half-life in
    expect(agent.getDiagnostics().learningRate).toBeCloseTo(alpha * 0.5, 5)

    const snap = agent.serialize()
    const restored = NTupleAgent.restore(snap.metaJson, snap.buffers)
    restored.refreshSchedule(100)
    expect(restored.getDiagnostics().learningRate).toBeCloseTo(alpha * 0.5, 5)
  })
})
