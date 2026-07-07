import { describe, expect, it } from 'vitest'
import { newBoard, maxExp, type Board } from '../src/engine/board'
import { applyMove, isGameOver, legalMask, spawnTile, startBoard } from '../src/engine/fastEngine'
import { splitmix32 } from '../src/engine/rng'
import type { Agent } from '../src/agents/types'
import { SYMMETRY_MAPS, applySymmetry, expandPattern } from '../src/agents/ntuple/symmetry'
import { NTupleNetwork } from '../src/agents/ntuple/network'
import { NTupleAgent } from '../src/agents/ntuple/tdLearner'
import { RandomAgent } from '../src/agents/randomAgent'
import type { NTupleConfig } from '../src/shared/types'

function playGame(agent: Agent, seed: number) {
  const board = newBoard()
  const after = newBoard()
  const next = newBoard()
  const rand = splitmix32(seed)
  startBoard(board, rand)
  let score = 0
  let moves = 0
  for (;;) {
    const mask = legalMask(board)
    if (mask === 0) break
    const ev = agent.evaluateMoves(board, mask, rand)
    const r = applyMove(board, ev.chosen, after)
    next.set(after)
    spawnTile(next, rand)
    const terminal = isGameOver(next)
    agent.observeTransition({
      state: board,
      action: ev.chosen,
      reward: r.score,
      afterstate: after,
      next,
      terminal,
    })
    board.set(next)
    score += r.score
    moves++
    if (terminal) break
  }
  const summary = { score, moves, maxExp: maxExp(board) }
  agent.onEpisodeEnd(summary)
  return summary
}

const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

describe('symmetry machinery', () => {
  it('produces 8 distinct dihedral maps', () => {
    const keys = new Set(SYMMETRY_MAPS.map((m) => m.join(',')))
    expect(keys.size).toBe(8)
    for (const m of SYMMETRY_MAPS) {
      expect([...m].sort((a, b) => a - b)).toEqual(Array.from({ length: 16 }, (_, i) => i))
    }
  })

  it('non-symmetric expansion is a single identity view', () => {
    expect(expandPattern([0, 1, 4, 5], false)).toEqual([[0, 1, 4, 5]])
  })

  it('V(board) is identical under all 8 board symmetries (shared weights)', () => {
    const net = new NTupleNetwork([[0, 1, 2, 3], [0, 1, 4, 5]], true)
    const rand = splitmix32(99)
    for (const t of net.tables) {
      for (let i = 0; i < t.length; i++) t[i] = rand() * 2 - 1
    }
    const board = newBoard()
    const transformed = newBoard()
    for (let trial = 0; trial < 50; trial++) {
      for (let i = 0; i < 16; i++) board[i] = (rand() * 16) | 0
      const base = net.value(board)
      for (const map of SYMMETRY_MAPS) {
        applySymmetry(board, map, transformed as Board)
        expect(net.value(transformed)).toBeCloseTo(base, 3)
      }
    }
  })
})

describe('TD(0) afterstate learning — the headline claim', () => {
  const config: NTupleConfig = {
    kind: 'ntuple',
    preset: 'starter',
    alpha: 1.0,
    tc: true,
    optimisticInit: 0,
    planningDepth: 1,
    gradeStrength: 1,
  }

  it('a fresh model plays at the random baseline, then genuinely learns', () => {
    // Random baseline for reference.
    const random = new RandomAgent()
    const baselineScores: number[] = []
    for (let g = 0; g < 200; g++) baselineScores.push(playGame(random, 5000 + g).score)
    const baseline = avg(baselineScores)
    expect(baseline).toBeGreaterThan(400)
    expect(baseline).toBeLessThan(2200)

    // Train a fresh starter n-tuple model.
    const agent = new NTupleAgent(config)
    const scores: number[] = []
    let bestExp = 0
    for (let g = 0; g < 1500; g++) {
      const s = playGame(agent, 42_000 + g)
      scores.push(s.score)
      if (s.maxExp > bestExp) bestExp = s.maxExp
    }
    const early = avg(scores.slice(0, 100))
    const late = avg(scores.slice(-300))
    // Genuine improvement: well past the random baseline and past its own start.
    expect(late).toBeGreaterThan(baseline * 2.5)
    expect(late).toBeGreaterThan(early * 1.5)
    expect(late).toBeGreaterThan(3000)
    // Should have discovered at least a 256 tile along the way.
    expect(bestExp).toBeGreaterThanOrEqual(8)
  }, 120_000)

  it('serialize → restore is lossless and keeps playing at the same level', () => {
    const agent = new NTupleAgent(config)
    for (let g = 0; g < 300; g++) playGame(agent, 7000 + g)
    const snap = agent.serialize()
    const restored = NTupleAgent.restore(snap.metaJson, snap.buffers)

    // Identical evaluations on random boards.
    const rand = splitmix32(3)
    const board = newBoard()
    for (let trial = 0; trial < 20; trial++) {
      for (let i = 0; i < 16; i++) board[i] = rand() < 0.4 ? 0 : (rand() * 10) | 0
      const mask = legalMask(board)
      if (mask === 0) continue
      const a = agent.evaluateMoves(board, mask, rand)
      const b = restored.evaluateMoves(board, mask, rand)
      expect(b.values).toEqual(a.values)
      expect(b.chosen).toBe(a.chosen)
    }
  }, 60_000)

  it('grading a move as Bad lowers that action value; Good raises it', () => {
    const agent = new NTupleAgent(config)
    for (let g = 0; g < 200; g++) playGame(agent, 11_000 + g)

    const board = newBoard()
    startBoard(board, splitmix32(1234))
    const mask = legalMask(board)
    const rand = splitmix32(77)
    const before = agent.evaluateMoves(board, mask, rand)
    const dir = before.chosen
    const after = newBoard()
    const next = newBoard()
    const r = applyMove(board, dir, after)
    next.set(after)
    spawnTile(next, splitmix32(55))

    const graded = (grade: -1 | 1, times: number) => {
      for (let i = 0; i < times; i++) {
        agent.observeTransition({
          state: board,
          action: dir,
          reward: r.score,
          afterstate: after,
          next,
          terminal: false,
          grade,
        })
        agent.onEpisodeEnd({ score: 0, moves: 1, maxExp: maxExp(next) })
      }
      return agent.evaluateMoves(board, mask, rand).values[dir]
    }

    const afterBad = graded(-1, 10)
    expect(afterBad).toBeLessThan(before.values[dir])
    const afterGood = graded(1, 30)
    expect(afterGood).toBeGreaterThan(afterBad)
  })
})
