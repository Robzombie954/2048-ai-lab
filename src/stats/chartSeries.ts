import { BUCKET_SIZE, type StatBucket } from '../shared/types'

function finite(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n)
}

function clampBucketN(n: number): number {
  return Math.max(0, Math.min(BUCKET_SIZE, Math.trunc(n)))
}

function cleanHist(hist: unknown): number[] {
  const out = new Array(16).fill(0)
  if (!Array.isArray(hist)) return out
  for (let i = 0; i < Math.min(16, hist.length); i++) {
    const v = hist[i]
    out[i] = finite(v) && v > 0 ? Math.trunc(v) : 0
  }
  return out
}

export function cleanBucket(raw: StatBucket | null | undefined): StatBucket | null {
  if (!raw || !finite(raw.bucket) || !finite(raw.n)) return null
  const bucket = Math.trunc(raw.bucket)
  const n = clampBucketN(raw.n)
  if (bucket < 0 || n <= 0) return null
  const scoreSum = finite(raw.scoreSum) ? raw.scoreSum : 0
  const scoreSqSum = finite(raw.scoreSqSum) ? Math.max(0, raw.scoreSqSum) : 0
  const movesSum = finite(raw.movesSum) ? Math.max(0, raw.movesSum) : 0
  return {
    bucket,
    n,
    scoreSum,
    scoreSqSum,
    scoreMin: finite(raw.scoreMin) ? raw.scoreMin : 0,
    scoreMax: finite(raw.scoreMax) ? raw.scoreMax : 0,
    movesSum,
    maxTileHist: cleanHist(raw.maxTileHist),
    tdErrAvg: finite(raw.tdErrAvg) ? raw.tdErrAvg : 0,
    epsAvg: finite(raw.epsAvg) ? raw.epsAvg : 0,
    lrAvg: finite(raw.lrAvg) ? raw.lrAvg : 0,
    wallMs: finite(raw.wallMs) ? Math.max(0, raw.wallMs) : 0,
    endTs: finite(raw.endTs) ? raw.endTs : 0,
  }
}

export function normalizeBuckets(raw: readonly (StatBucket | null | undefined)[]): StatBucket[] {
  const byBucket = new Map<number, StatBucket>()
  for (const item of raw) {
    const bucket = cleanBucket(item)
    if (!bucket) continue
    const prev = byBucket.get(bucket.bucket)
    if (!prev || bucket.n > prev.n || (bucket.n === prev.n && bucket.endTs >= prev.endTs)) {
      byBucket.set(bucket.bucket, bucket)
    }
  }
  return Array.from(byBucket.values()).sort((a, b) => a.bucket - b.bucket)
}

export function bucketX(b: StatBucket): number {
  return b.bucket * BUCKET_SIZE + b.n
}

export function safeMean(sum: number, n: number): number {
  return n > 0 && Number.isFinite(sum) ? sum / n : 0
}

export function safeBucketMean(b: StatBucket): number {
  return safeMean(b.scoreSum, b.n)
}

export function safeMovesPerGame(b: StatBucket): number {
  return safeMean(b.movesSum, b.n)
}

export function finiteOrNull(v: number): number | null {
  return Number.isFinite(v) ? v : null
}

/** Standard deviation of scores within a bucket, from the stored sum-of-squares. */
export function bucketStd(b: StatBucket): number {
  if (b.n <= 0) return 0
  const mean = b.scoreSum / b.n
  const variance = b.scoreSqSum / b.n - mean * mean
  return variance > 0 && Number.isFinite(variance) ? Math.sqrt(variance) : 0
}

export type TrendLabel = 'rising' | 'converging' | 'converged' | 'falling'

export interface TrajectorySummary {
  slopePerGame: number
  deltaPerThousandGames: number
  /** Mean score over the summary window — used for the relative flat threshold. */
  currentAvg: number
  label: TrendLabel
}

function finitePoint(x: number, y: number): boolean {
  return Number.isFinite(x) && Number.isFinite(y)
}

function regression(points: readonly (readonly [number, number])[]): { slope: number; intercept: number } | null {
  if (points.length < 2) return null
  let sx = 0
  let sy = 0
  let sxx = 0
  let sxy = 0
  for (const [x, y] of points) {
    sx += x
    sy += y
    sxx += x * x
    sxy += x * y
  }
  const n = points.length
  const denom = n * sxx - sx * sx
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-9) return null
  const slope = (n * sxy - sx * sy) / denom
  const intercept = (sy - slope * sx) / n
  return { slope, intercept }
}

export function rollingTrajectory(
  xs: readonly number[],
  ys: readonly number[],
  windowSize = 12,
): (number | null)[] {
  const out: (number | null)[] = []
  const window: [number, number][] = []
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]
    const y = ys[i]
    if (finitePoint(x, y)) {
      window.push([x, y])
      if (window.length > windowSize) window.shift()
    }
    const fit = regression(window)
    out.push(fit ? fit.slope * x + fit.intercept : finitePoint(x, y) ? y : null)
  }
  return out
}

export function summarizeTrajectory(
  xs: readonly number[],
  ys: readonly number[],
  windowSize = 12,
): TrajectorySummary | null {
  const points: [number, number][] = []
  let ySum = 0
  for (let i = Math.max(0, xs.length - windowSize); i < xs.length; i++) {
    if (finitePoint(xs[i], ys[i])) {
      points.push([xs[i], ys[i]])
      ySum += ys[i]
    }
  }
  const fit = regression(points)
  if (!fit) return null
  const currentAvg = points.length ? ySum / points.length : 0
  const deltaPerThousandGames = fit.slope * 1000
  // Flat is judged RELATIVE to the current score level: a model averaging 35k
  // that gains +300/1k games is essentially converged, not "rising". A fixed
  // threshold mislabels big models as flat and tiny ones as rising.
  const rel = Math.max(250, 0.02 * currentAvg)
  let label: TrendLabel
  if (deltaPerThousandGames < -rel) label = 'falling'
  else if (deltaPerThousandGames <= rel) label = 'converged'
  else if (deltaPerThousandGames <= rel * 3) label = 'converging'
  else label = 'rising'
  return { slopePerGame: fit.slope, deltaPerThousandGames, currentAvg, label }
}

export function projectedTrajectory(
  xs: readonly number[],
  ys: readonly number[],
  windowSize = 12,
  projectedBuckets = 8,
): { x: number[]; y: (number | null)[] } {
  if (xs.length === 0) return { x: [], y: [] }
  const points: [number, number][] = []
  for (let i = Math.max(0, xs.length - windowSize); i < xs.length; i++) {
    if (finitePoint(xs[i], ys[i])) points.push([xs[i], ys[i]])
  }
  const fit = regression(points)
  const lastX = xs[xs.length - 1]
  if (!fit || !Number.isFinite(lastX)) return { x: [lastX], y: [finiteOrNull(ys[ys.length - 1])] }
  const x = [lastX]
  const y: (number | null)[] = [fit.slope * lastX + fit.intercept]
  for (let i = 1; i <= projectedBuckets; i++) {
    const nextX = lastX + BUCKET_SIZE * i
    x.push(nextX)
    y.push(Math.max(0, fit.slope * nextX + fit.intercept))
  }
  return { x, y }
}

