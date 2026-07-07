// Per-100-game aggregate buckets: the permanent, bounded-size stat storage
// that keeps charts instant even after millions of games (1M games ≈ 10k
// buckets ≈ 2MB).
import { BUCKET_SIZE, type GameSummary, type StatBucket } from '../shared/types'

interface Accum {
  bucket: number
  n: number
  scoreSum: number
  scoreSqSum: number
  scoreMin: number
  scoreMax: number
  movesSum: number
  maxTileHist: number[]
  tdErrSum: number
  epsSum: number
  lrSum: number
  wallMs: number
}

function freshAccum(bucket: number): Accum {
  return {
    bucket,
    n: 0,
    scoreSum: 0,
    scoreSqSum: 0,
    scoreMin: Infinity,
    scoreMax: -Infinity,
    movesSum: 0,
    maxTileHist: new Array(16).fill(0),
    tdErrSum: 0,
    epsSum: 0,
    lrSum: 0,
    wallMs: 0,
  }
}

function finalize(a: Accum, endTs: number): StatBucket {
  return {
    bucket: a.bucket,
    n: a.n,
    scoreSum: a.scoreSum,
    scoreSqSum: a.scoreSqSum,
    scoreMin: a.n ? a.scoreMin : 0,
    scoreMax: a.n ? a.scoreMax : 0,
    movesSum: a.movesSum,
    maxTileHist: [...a.maxTileHist],
    tdErrAvg: a.n ? a.tdErrSum / a.n : 0,
    epsAvg: a.n ? a.epsSum / a.n : 0,
    lrAvg: a.n ? a.lrSum / a.n : 0,
    wallMs: a.wallMs,
    endTs,
  }
}

export class BucketAggregator {
  private current: Accum | null = null

  /** Rebuild a partial bucket from raw game summaries (resume mid-bucket). */
  seed(summaries: GameSummary[], tdErr: number, eps: number, lr: number): void {
    for (const s of summaries) {
      this.addGame(s, tdErr, eps, lr, 0, 0)
    }
  }

  /** Returns the completed bucket when this game closes one. */
  addGame(
    s: GameSummary,
    tdErr: number,
    eps: number,
    lr: number,
    wallMsDelta: number,
    endTs = 0,
  ): StatBucket | null {
    const bucket = Math.floor(s.game / BUCKET_SIZE)
    if (!this.current || this.current.bucket !== bucket) {
      this.current = freshAccum(bucket)
    }
    const a = this.current
    a.n++
    a.scoreSum += s.score
    a.scoreSqSum += s.score * s.score
    if (s.score < a.scoreMin) a.scoreMin = s.score
    if (s.score > a.scoreMax) a.scoreMax = s.score
    a.movesSum += s.moves
    a.maxTileHist[Math.min(s.maxExp, 15)]++
    a.tdErrSum += tdErr
    a.epsSum += eps
    a.lrSum += lr
    a.wallMs += wallMsDelta
    if ((s.game + 1) % BUCKET_SIZE === 0) {
      const done = finalize(a, endTs)
      this.current = null
      return done
    }
    return null
  }

  /** Live view of the in-progress bucket (for charts' rightmost point). */
  partial(endTs = 0): StatBucket | null {
    return this.current && this.current.n > 0 ? finalize(this.current, endTs) : null
  }

  reset(): void {
    this.current = null
  }
}

/** Chart helpers shared by UI. */
export function bucketMean(b: StatBucket): number {
  return b.n ? b.scoreSum / b.n : 0
}

export function bucketStdDev(b: StatBucket): number {
  if (!b.n) return 0
  const mean = b.scoreSum / b.n
  return Math.sqrt(Math.max(0, b.scoreSqSum / b.n - mean * mean))
}
