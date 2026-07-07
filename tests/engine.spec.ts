import { describe, expect, it } from 'vitest'
import { slideLine, UP, RIGHT, DOWN, LEFT, DIRS } from '../src/engine/rules'
import { boardFromValues, boardToValues, newBoard, cloneBoard } from '../src/engine/board'
import { LEFT_RESULT, LEFT_SCORE, RIGHT_RESULT, RIGHT_SCORE, packRow } from '../src/engine/luts'
import {
  applyMove,
  legalMask,
  isGameOver,
  spawnTile,
  startBoard,
} from '../src/engine/fastEngine'
import { slideLineWithIds, TraceGame } from '../src/engine/traceEngine'
import { splitmix32 } from '../src/engine/rng'

function unpackRow(code: number): number[] {
  return [code >>> 12, (code >>> 8) & 0xf, (code >>> 4) & 0xf, code & 0xf]
}

describe('canonical rules — oracle cases (tile values)', () => {
  const cases: { line: number[]; expected: number[]; score: number }[] = [
    { line: [2, 2, 0, 0], expected: [4, 0, 0, 0], score: 4 },
    { line: [2, 2, 4, 0], expected: [4, 4, 0, 0], score: 4 }, // no cascade
    { line: [2, 2, 2, 2], expected: [4, 4, 0, 0], score: 8 },
    { line: [2, 2, 2, 0], expected: [4, 2, 0, 0], score: 4 }, // edge-side merges first
    { line: [4, 2, 2, 0], expected: [4, 4, 0, 0], score: 4 }, // one merge per tile
    { line: [0, 2, 0, 2], expected: [4, 0, 0, 0], score: 4 },
    { line: [2, 0, 0, 2], expected: [4, 0, 0, 0], score: 4 },
    { line: [4, 4, 8, 8], expected: [8, 16, 0, 0], score: 24 },
    { line: [0, 0, 0, 2], expected: [2, 0, 0, 0], score: 0 },
    { line: [2, 4, 2, 4], expected: [2, 4, 2, 4], score: 0 }, // no move
    { line: [0, 0, 0, 0], expected: [0, 0, 0, 0], score: 0 },
  ]
  for (const c of cases) {
    it(`[${c.line}] → [${c.expected}] (+${c.score})`, () => {
      const exps = c.line.map((v) => (v ? Math.log2(v) : 0))
      const r = slideLine(exps)
      expect(r.out.map((e) => (e ? 1 << e : 0))).toEqual(c.expected)
      expect(r.score).toBe(c.score)
      expect(r.moved).toBe(c.line.join() !== c.expected.join())
    })
  }
})

describe('trace engine ≡ canonical rules — exhaustive, all 65,536 rows', () => {
  it('slideLineWithIds matches slideLine on every possible row', () => {
    for (let row = 0; row < 0x10000; row++) {
      const cells = unpackRow(row)
      const canonical = slideLine(cells)
      const entries = []
      for (let p = 0; p < 4; p++) {
        if (cells[p] !== 0) entries.push({ id: p + 1, exp: cells[p], pos: p })
      }
      const traced = slideLineWithIds(entries)
      const tracedOut = [0, 0, 0, 0]
      for (const pl of traced.placed) tracedOut[pl.slot] = pl.exp
      if (
        tracedOut[0] !== canonical.out[0] ||
        tracedOut[1] !== canonical.out[1] ||
        tracedOut[2] !== canonical.out[2] ||
        tracedOut[3] !== canonical.out[3] ||
        traced.score !== canonical.score
      ) {
        throw new Error(
          `row ${row.toString(16)}: trace [${tracedOut}] +${traced.score} != canonical [${canonical.out}] +${canonical.score}`,
        )
      }
    }
  })

  it('LUT row tables match slideLine on every possible row (left & right)', () => {
    for (let row = 0; row < 0x10000; row++) {
      const cells = unpackRow(row)
      const left = slideLine(cells)
      expect(LEFT_RESULT[row]).toBe(packRow(left.out[0], left.out[1], left.out[2], left.out[3]))
      if (LEFT_SCORE[row] !== left.score) throw new Error(`left score mismatch row ${row}`)
      const right = slideLine([cells[3], cells[2], cells[1], cells[0]])
      if (RIGHT_RESULT[row] !== packRow(right.out[3], right.out[2], right.out[1], right.out[0]))
        throw new Error(`right result mismatch row ${row}`)
      if (RIGHT_SCORE[row] !== right.score) throw new Error(`right score mismatch row ${row}`)
    }
  })
})

describe('fast engine ⇄ trace engine — full-game lockstep', () => {
  it('100 seeded games play identically through both engines', () => {
    for (let g = 0; g < 100; g++) {
      const seed = 1000 + g
      const game = new TraceGame(seed)
      // Shadow the trace game with the fast engine on a separate board.
      const shadow = cloneBoard(game.board)
      const scratch = newBoard()
      const moveRand = splitmix32(seed ^ 0x5eed)
      let shadowScore = 0
      let guard = 0
      while (!game.over && guard++ < 5000) {
        const mask = legalMask(shadow)
        expect(mask).toBe(game.legalMask())
        const legal = DIRS.filter((d) => mask & (1 << d))
        const dir = legal[(moveRand() * legal.length) | 0]
        const events = game.move(dir) // trace game spawns from its own rng
        expect(events).not.toBeNull()
        const r = applyMove(shadow, dir, scratch)
        expect(r.moved).toBe(true)
        shadowScore += r.score
        expect(r.score).toBe(events!.scoreGained)
        shadow.set(scratch)
        // Replay the trace game's spawn onto the shadow board.
        expect(events!.spawn).not.toBeNull()
        expect(shadow[events!.spawn!.idx]).toBe(0)
        shadow[events!.spawn!.idx] = events!.spawn!.exp
        expect(Array.from(shadow)).toEqual(Array.from(game.board))
        expect(shadowScore).toBe(game.score)
      }
      expect(game.over).toBe(true)
      expect(isGameOver(shadow)).toBe(true)
    }
  })
})

describe('spawning', () => {
  it('spawns 2 vs 4 at 0.9/0.1 within ±1% over 100k draws', () => {
    const rand = splitmix32(42)
    const board = newBoard()
    let twos = 0
    const n = 100_000
    for (let i = 0; i < n; i++) {
      board.fill(0)
      const s = spawnTile(board, rand)
      if (s.exp === 1) twos++
    }
    expect(twos / n).toBeGreaterThan(0.89)
    expect(twos / n).toBeLessThan(0.91)
  })

  it('spawns only into empty cells, uniformly', () => {
    const rand = splitmix32(7)
    const counts = new Array(16).fill(0)
    for (let i = 0; i < 32_000; i++) {
      const board = boardFromValues([2, 0, 0, 4, 0, 8, 0, 0, 0, 0, 16, 0, 0, 2, 0, 0])
      const s = spawnTile(board, rand)
      expect([0, 3, 5, 10, 13].includes(s.idx)).toBe(false)
      counts[s.idx]++
    }
    const emptyIdx = [1, 2, 4, 6, 7, 8, 9, 11, 12, 14, 15]
    const expected = 32_000 / emptyIdx.length
    for (const i of emptyIdx) {
      expect(counts[i]).toBeGreaterThan(expected * 0.85)
      expect(counts[i]).toBeLessThan(expected * 1.15)
    }
  })

  it('startBoard yields two tiles', () => {
    const board = newBoard()
    startBoard(board, splitmix32(1))
    const tiles = Array.from(board).filter((e) => e !== 0)
    expect(tiles.length).toBe(2)
  })
})

describe('game over & legality', () => {
  it('detects game over only when no move changes the board', () => {
    const stuck = boardFromValues([2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2])
    expect(isGameOver(stuck)).toBe(true)
    const mergeable = boardFromValues([2, 2, 4, 8, 4, 8, 16, 2, 2, 4, 8, 16, 4, 8, 16, 2])
    expect(isGameOver(mergeable)).toBe(false)
  })

  it('full board with a merge available is not over', () => {
    const b = boardFromValues([2, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2, 4, 8, 16, 32])
    expect(isGameOver(b)).toBe(false)
    expect(legalMask(b) & ((1 << LEFT) | (1 << RIGHT))).toBeTruthy()
    expect(legalMask(b) & ((1 << UP) | (1 << DOWN))).toBeFalsy()
  })

  it('boardToValues round-trips', () => {
    const values = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 0, 2, 4]
    expect(boardToValues(boardFromValues(values))).toEqual(values)
  })
})
