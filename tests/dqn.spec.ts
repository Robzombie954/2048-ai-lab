import { describe, expect, it } from 'vitest'
import { newBoard, maxExp } from '../src/engine/board'
import { applyMove, isGameOver, legalMask, spawnTile, startBoard } from '../src/engine/fastEngine'
import { splitmix32 } from '../src/engine/rng'
import { MLP } from '../src/agents/dqn/mlp'
import { ReplayBuffer } from '../src/agents/dqn/replay'
import { DQNAgent } from '../src/agents/dqn/dqnAgent'
import { defaultDQNConfig } from '../src/agents/factory'

describe('MLP — numerical gradient check', () => {
  it('analytic gradients match central differences', () => {
    const rand = splitmix32(1)
    const net = new MLP([6, 8, 4], rand)
    const acts = net.makeActs()
    const x = Float32Array.from({ length: 6 }, () => rand() * 2 - 1)
    const action = 2
    const target = 0.7

    const loss = () => {
      acts[0].set(x)
      const q = net.forward(acts)
      const err = q[action] - target
      // Huber, delta = 1
      return Math.abs(err) <= 1 ? 0.5 * err * err : Math.abs(err) - 0.5
    }

    // Analytic gradients.
    acts[0].set(x)
    const q = net.forward(acts)
    const err = q[action] - target
    const dOut = new Float32Array(4)
    dOut[action] = Math.max(-1, Math.min(1, err))
    net.zeroGrads()
    net.backward(acts, dOut)
    const gW = (net as unknown as { gW: Float32Array[] }).gW

    // Numerical check on a sample of weights from each layer.
    const eps = 1e-3
    let checked = 0
    for (let l = 0; l < net.W.length; l++) {
      for (let k = 0; k < 12; k++) {
        const i = (rand() * net.W[l].length) | 0
        const orig = net.W[l][i]
        net.W[l][i] = orig + eps
        const lp = loss()
        net.W[l][i] = orig - eps
        const lm = loss()
        net.W[l][i] = orig
        const numeric = (lp - lm) / (2 * eps)
        const analytic = gW[l][i]
        const scale = Math.max(Math.abs(numeric), Math.abs(analytic), 1e-4)
        expect(Math.abs(numeric - analytic) / scale).toBeLessThan(0.05)
        checked++
      }
    }
    expect(checked).toBeGreaterThan(20)
  })
})

describe('replay buffer', () => {
  it('ring-wraps and returns the stored transitions', () => {
    const buf = new ReplayBuffer(8)
    const b = newBoard()
    for (let i = 0; i < 12; i++) {
      b.fill(0)
      b[0] = i % 16
      buf.push(b, i % 4, i, b, i % 2 === 0, false)
    }
    expect(buf.size).toBe(8)
    // Slot 0 was overwritten by push #8.
    expect(buf.stateAt(0)[0]).toBe(8)
    expect(buf.rewardAt(0)).toBe(8)
    expect(buf.actionAt(3)).toBe(11 % 4)
  })

  it('boosts graded samples', () => {
    const buf = new ReplayBuffer(1000)
    const b = newBoard()
    for (let i = 0; i < 1000; i++) buf.push(b, 0, 0, b, false, i < 10) // 1% graded
    const rand = splitmix32(5)
    let graded = 0
    const draws = 20_000
    for (let i = 0; i < draws; i++) {
      if (buf.sampleIndex(rand) < 10) graded++
    }
    const rate = graded / draws
    expect(rate).toBeGreaterThan(0.025) // ≳2.5× the uniform 1%
  })
})

describe('DQN agent', () => {
  it('plays full games, learns without NaN, epsilon decays', () => {
    const config = { ...defaultDQNConfig(), epsDecayMoves: 500, replaySize: 5000 }
    const agent = new DQNAgent(config, 7)
    const board = newBoard()
    const after = newBoard()
    const next = newBoard()
    const rand = splitmix32(31337)
    const eps0 = agent.epsilon()
    for (let g = 0; g < 25; g++) {
      startBoard(board, rand)
      for (;;) {
        const mask = legalMask(board)
        if (mask === 0) break
        const ev = agent.evaluateMoves(board, mask, rand)
        expect(Number.isFinite(ev.values[ev.chosen])).toBe(true)
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
        if (terminal) break
      }
      agent.onEpisodeEnd({ score: 0, moves: 0, maxExp: maxExp(board) })
    }
    expect(agent.epsilon()).toBeLessThan(eps0)
    const d = agent.getDiagnostics()
    expect(Number.isFinite(d.meanAbsTdError)).toBe(true)
    expect(d.paramCount).toBeGreaterThan(40_000)
  }, 60_000)

  it('serialize → restore reproduces evaluations exactly', () => {
    const agent = new DQNAgent({ ...defaultDQNConfig(), hidden: [32, 16] }, 11)
    const board = newBoard()
    const rand = splitmix32(2)
    startBoard(board, rand)
    const snap = agent.serialize()
    const restored = DQNAgent.restore(snap.metaJson, snap.buffers)
    const mask = legalMask(board)
    // Compare greedy Q-values (fixed rand to dodge epsilon).
    const one = () => 0.999999
    const a = agent.evaluateMoves(board, mask, one)
    const b = restored.evaluateMoves(board, mask, one)
    expect(b.values).toEqual(a.values)
  })
})

