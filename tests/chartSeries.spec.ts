import { describe, expect, it } from 'vitest'
import { bucketX, normalizeBuckets, projectedTrajectory, rollingTrajectory, safeBucketMean, summarizeTrajectory } from '../src/stats/chartSeries'
import type { StatBucket } from '../src/shared/types'

const bucket = (patch: Partial<StatBucket>): StatBucket => ({
  bucket: 0,
  n: 100,
  scoreSum: 1000,
  scoreSqSum: 10000,
  scoreMin: 1,
  scoreMax: 100,
  movesSum: 500,
  maxTileHist: new Array(16).fill(0),
  tdErrAvg: 1,
  epsAvg: 0,
  lrAvg: 0.1,
  wallMs: 100,
  endTs: 1,
  ...patch,
})

describe('chart bucket normalization', () => {
  it('sorts, deduplicates, and keeps x values strictly increasing', () => {
    const all = normalizeBuckets([
      bucket({ bucket: 2, n: 50, scoreSum: 5000, endTs: 2 }),
      bucket({ bucket: 0, n: 100, scoreSum: 1000, endTs: 1 }),
      bucket({ bucket: 1, n: 80, scoreSum: 8000, endTs: 1 }),
      bucket({ bucket: 1, n: 100, scoreSum: 12000, endTs: 2 }),
      bucket({ bucket: 2, n: 20, scoreSum: 2000, endTs: 3 }),
    ])

    expect(all.map((b) => b.bucket)).toEqual([0, 1, 2])
    expect(all.map((b) => b.n)).toEqual([100, 100, 50])
    expect(all.map(safeBucketMean)).toEqual([10, 120, 100])
    const xs = all.map(bucketX)
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1])
  })

  it('drops unusable buckets and sanitizes non-finite chart values', () => {
    const all = normalizeBuckets([
      bucket({ bucket: -1, n: 10 }),
      bucket({ bucket: 0, n: 0 }),
      bucket({ bucket: 1, n: Number.NaN }),
      bucket({ bucket: 2, n: 20, scoreSum: Number.NaN, scoreMax: Number.POSITIVE_INFINITY, movesSum: Number.NEGATIVE_INFINITY }),
    ])

    expect(all).toHaveLength(1)
    expect(all[0].bucket).toBe(2)
    expect(all[0].scoreSum).toBe(0)
    expect(all[0].scoreMax).toBe(0)
    expect(all[0].movesSum).toBe(0)
  })
  it('summarizes a rising score trajectory and projects future points', () => {
    const xs = [100, 200, 300, 400, 500]
    const ys = [1000, 1200, 1400, 1600, 1800]
    const summary = summarizeTrajectory(xs, ys, 5)
    expect(summary?.label).toBe('rising')
    expect(summary?.deltaPerThousandGames).toBeCloseTo(2000, 4)

    const trend = rollingTrajectory(xs, ys, 3)
    expect(trend).toHaveLength(xs.length)
    expect(trend[trend.length - 1]).toBeCloseTo(1800, 4)

    const projection = projectedTrajectory(xs, ys, 5, 2)
    expect(projection.x).toEqual([500, 600, 700])
    expect(projection.y[2]).toBeCloseTo(2200, 4)
  })

  it('treats tiny trajectory changes as converged', () => {
    const summary = summarizeTrajectory([100, 200, 300, 400], [1000, 1005, 1010, 1015], 4)
    expect(summary?.label).toBe('converged')
  })
})

