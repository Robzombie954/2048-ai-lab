import { describe, expect, it } from 'vitest'
import { buildExport, parseExport } from '../src/persistence/exportImport'
import { newModelDoc } from '../src/persistence/modelStore'
import { BucketAggregator } from '../src/stats/buckets'
import { MilestoneTracker } from '../src/stats/milestones'
import { defaultNTupleConfig } from '../src/agents/factory'
import { NTupleAgent } from '../src/agents/ntuple/tdLearner'
import { BUCKET_SIZE, type StatBucket } from '../src/shared/types'

describe('.2048model export container', () => {
  it('round-trips byte-exactly', () => {
    const config = defaultNTupleConfig('starter')
    const agent = new NTupleAgent(config)
    const snap = agent.serialize()
    // Make weights non-trivial.
    new Float32Array(snap.buffers[0])[123] = 42.5
    const doc = newModelDoc('Test Model', config)
    doc.games = 777
    const buckets: StatBucket[] = [
      {
        bucket: 0,
        n: 100,
        scoreSum: 123456,
        scoreSqSum: 999,
        scoreMin: 12,
        scoreMax: 3456,
        movesSum: 10000,
        maxTileHist: new Array(16).fill(0),
        tdErrAvg: 1.5,
        epsAvg: 0,
        lrAvg: 1,
        wallMs: 60000,
        endTs: 1700000000000,
      },
    ]
    const data = buildExport(
      doc,
      {
        gamesAt: 777,
        movesAt: 88_000,
        trainMsAt: 3_600_000,
        rollingAvg: 5432.1,
        metaJson: snap.metaJson,
        buffers: snap.buffers,
      },
      buckets,
    )
    const parsed = parseExport(data)
    expect(parsed.header.model.name).toBe('Test Model')
    expect(parsed.header.checkpoint.gamesAt).toBe(777)
    expect(parsed.header.checkpoint.metaJson).toBe(snap.metaJson)
    expect(parsed.header.buckets).toEqual(JSON.parse(JSON.stringify(buckets)))
    expect(parsed.buffers.length).toBe(snap.buffers.length)
    for (let i = 0; i < snap.buffers.length; i++) {
      const a = new Uint8Array(parsed.buffers[i])
      const b = new Uint8Array(snap.buffers[i])
      expect(a.byteLength).toBe(b.byteLength)
      let mismatch = -1
      for (let j = 0; j < a.length; j++) {
        if (a[j] !== b[j]) {
          mismatch = j
          break
        }
      }
      expect(mismatch).toBe(-1)
    }
    // And the parsed buffers restore to an identical agent.
    const restored = NTupleAgent.restore(parsed.header.checkpoint.metaJson, parsed.buffers)
    expect(restored.net.tables[0][123]).toBeCloseTo(42.5)
  })

  it('rejects garbage', () => {
    expect(() => parseExport(new ArrayBuffer(4))).toThrow()
    const junk = new ArrayBuffer(64)
    new Uint8Array(junk).fill(65)
    expect(() => parseExport(junk)).toThrow(/magic/)
  })
})

describe('bucket aggregation', () => {
  it('closes a bucket every 100 games with correct sums', () => {
    const agg = new BucketAggregator()
    let completed: StatBucket | null = null
    for (let g = 0; g < BUCKET_SIZE; g++) {
      completed = agg.addGame(
        { game: g, score: g * 10, moves: 50, maxExp: 9, seed: 1 },
        0.5,
        0.1,
        1,
        100,
        123,
      )
      if (g < BUCKET_SIZE - 1) expect(completed).toBeNull()
    }
    expect(completed).not.toBeNull()
    expect(completed!.n).toBe(100)
    expect(completed!.scoreSum).toBe((99 * 100 * 10) / 2)
    expect(completed!.scoreMin).toBe(0)
    expect(completed!.scoreMax).toBe(990)
    expect(completed!.movesSum).toBe(5000)
    expect(completed!.maxTileHist[9]).toBe(100)
    expect(completed!.tdErrAvg).toBeCloseTo(0.5)
    expect(completed!.wallMs).toBe(10_000)
    expect(agg.partial()).toBeNull()
  })

  it('partial() reflects the in-progress bucket and seeding rebuilds it', () => {
    const agg = new BucketAggregator()
    agg.addGame({ game: 200, score: 500, moves: 10, maxExp: 5, seed: 1 }, 0, 0, 1, 0)
    agg.addGame({ game: 201, score: 700, moves: 12, maxExp: 6, seed: 1 }, 0, 0, 1, 0)
    const partial = agg.partial()
    expect(partial!.bucket).toBe(2)
    expect(partial!.n).toBe(2)
    expect(partial!.scoreSum).toBe(1200)

    const seeded = new BucketAggregator()
    seeded.seed(
      [
        { game: 200, score: 500, moves: 10, maxExp: 5, seed: 1 },
        { game: 201, score: 700, moves: 12, maxExp: 6, seed: 1 },
      ],
      0,
      0,
      1,
    )
    expect(seeded.partial()!.scoreSum).toBe(1200)
  })
})

describe('milestones', () => {
  it('fires once per tile exponent, starting at 128', () => {
    const t = new MilestoneTracker()
    expect(t.check(6, 1, 100, 0)).toBeNull() // 64 — below threshold
    const m = t.check(7, 2, 200, 5000)
    expect(m).not.toBeNull()
    expect(m!.exp).toBe(7)
    expect(t.check(7, 3, 300, 6000)).toBeNull() // already seen
    expect(t.check(9, 4, 400, 7000)!.exp).toBe(9)
  })
})
