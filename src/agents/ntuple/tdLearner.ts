// TD(0) afterstate learning (Szubert & Jaśkowski 2014): act greedily on
// r + V(afterstate); after seeing the next move, update the previous
// afterstate toward r' + V(next afterstate); toward 0 at terminal.
// Exploration comes from spawn stochasticity (plus optional optimistic init).
import type { Board } from '../../engine/board'
import { newBoard } from '../../engine/board'
import { DIRS } from '../../engine/rules'
import { applyMove } from '../../engine/fastEngine'
import type { Rand } from '../../engine/rng'
import type { AnnealConfig, NTupleConfig } from '../../shared/types'
import type {
  Agent,
  AgentDiagnostics,
  AgentSnapshot,
  MoveEvaluation,
  Transition,
} from '../types'
import { NTUPLE_PRESETS } from './patterns'
import { NTupleNetwork } from './network'
import { expectimaxValue } from './expectimax'
import { annealMultiplier } from './anneal'

export class NTupleAgent implements Agent {
  readonly kind = 'ntuple' as const
  planningDepth: number

  readonly net: NTupleNetwork
  private config: NTupleConfig
  private annealMult = 1
  private readonly tcE: Float32Array[] | null
  private readonly tcA: Float32Array[] | null

  private readonly afterScratch: Board[] = [newBoard(), newBoard(), newBoard(), newBoard()]
  private readonly prevAfter: Board = newBoard()
  private hasPrev = false
  private readonly gatherBuf: Int32Array

  private meanReward = 0 // EMA of per-move reward magnitude (scales grades)
  private tdErrEma = 0

  constructor(config: NTupleConfig, net?: NTupleNetwork) {
    this.config = config
    this.planningDepth = config.planningDepth
    const preset = NTUPLE_PRESETS[config.preset]
    this.net = net ?? new NTupleNetwork(preset.patterns, preset.symmetric, config.optimisticInit)
    this.gatherBuf = new Int32Array(this.net.totalViews)
    if (config.tc) {
      this.tcE = this.net.tables.map((t) => new Float32Array(t.length))
      this.tcA = this.net.tables.map((t) => new Float32Array(t.length))
    } else {
      this.tcE = null
      this.tcA = null
    }
  }

  evaluateMoves(state: Board, mask: number, _rand: Rand): MoveEvaluation {
    const values: [number, number, number, number] = [-Infinity, -Infinity, -Infinity, -Infinity]
    let best = -Infinity
    let chosen = 0
    for (const dir of DIRS) {
      if (!(mask & (1 << dir))) continue
      const after = this.afterScratch[dir]
      const r = applyMove(state, dir, after)
      const v =
        r.score +
        (this.planningDepth > 1
          ? expectimaxValue(this.net, after, this.planningDepth - 1)
          : this.net.value(after))
      values[dir] = v
      if (v > best) {
        best = v
        chosen = dir
      }
    }
    return { values, chosen: chosen as MoveEvaluation['chosen'], exploring: false }
  }

  observeTransition(t: Transition): void {
    // A manual grade is a genuine value-function update on the graded move's
    // afterstate, scaled to the model's current reward magnitude.
    if (t.grade) {
      const nudge = this.config.gradeStrength * t.grade * Math.max(this.meanReward, 4)
      this.updateToward(t.afterstate, this.net.value(t.afterstate) + nudge)
    }
    // Learning always evaluates the plain network (depth 1), independent of
    // any inference-time planning depth.
    const target = t.reward + this.net.value(t.afterstate)
    if (this.hasPrev) this.updateToward(this.prevAfter, target)
    if (t.terminal) {
      this.updateToward(t.afterstate, 0)
      this.hasPrev = false
    } else {
      this.prevAfter.set(t.afterstate)
      this.hasPrev = true
    }
    this.meanReward += 0.001 * (t.reward - this.meanReward)
  }

  onEpisodeEnd(_summary?: unknown): void {
    this.hasPrev = false
  }

  /** Per-game hook: recompute the annealing multiplier from games seen. */
  refreshSchedule(gamesSeen: number): void {
    this.annealMult = annealMultiplier(gamesSeen, this.config.anneal)
  }

  /** Enable/disable annealing at runtime (persists via config in serialize). */
  setAnneal(anneal: AnnealConfig | null, gamesSeen: number): void {
    this.config = { ...this.config, anneal: anneal ?? undefined }
    this.refreshSchedule(gamesSeen)
  }

  private updateToward(board: Board, target: number): void {
    const delta = target - this.net.value(board)
    this.tdErrEma += 0.001 * (Math.abs(delta) - this.tdErrEma)
    this.net.gatherIndices(board, this.gatherBuf)
    const per = (this.config.alpha * this.annealMult * delta) / this.net.totalViews
    const absDelta = Math.abs(delta)
    for (let vi = 0; vi < this.net.totalViews; vi++) {
      const p = this.net.viewPattern[vi]
      const idx = this.gatherBuf[vi]
      if (this.tcE && this.tcA) {
        const e = this.tcE[p][idx]
        const a = this.tcA[p][idx]
        const rate = a === 0 ? 1 : Math.abs(e) / a
        this.net.tables[p][idx] += per * rate
        this.tcE[p][idx] = e + delta
        this.tcA[p][idx] = a + absDelta
      } else {
        this.net.tables[p][idx] += per
      }
    }
  }

  serialize(): AgentSnapshot {
    const buffers: ArrayBuffer[] = []
    for (const t of this.net.tables) buffers.push(t.buffer.slice(0) as ArrayBuffer)
    if (this.tcE && this.tcA) {
      for (const t of this.tcE) buffers.push(t.buffer.slice(0) as ArrayBuffer)
      for (const t of this.tcA) buffers.push(t.buffer.slice(0) as ArrayBuffer)
    }
    return {
      metaJson: JSON.stringify({
        version: 1,
        config: this.config,
        meanReward: this.meanReward,
        tdErrEma: this.tdErrEma,
      }),
      buffers,
    }
  }

  static restore(metaJson: string, buffers: ArrayBuffer[]): NTupleAgent {
    const meta = JSON.parse(metaJson) as {
      config: NTupleConfig
      meanReward: number
      tdErrEma: number
    }
    const agent = new NTupleAgent(meta.config)
    const nTables = agent.net.tables.length
    const expected = meta.config.tc ? nTables * 3 : nTables
    if (buffers.length !== expected) {
      throw new Error(`checkpoint has ${buffers.length} buffers, expected ${expected}`)
    }
    for (let i = 0; i < nTables; i++) {
      agent.net.tables[i].set(new Float32Array(buffers[i]))
    }
    if (meta.config.tc && agent.tcE && agent.tcA) {
      for (let i = 0; i < nTables; i++) {
        agent.tcE[i].set(new Float32Array(buffers[nTables + i]))
        agent.tcA[i].set(new Float32Array(buffers[nTables * 2 + i]))
      }
    }
    agent.meanReward = meta.meanReward
    agent.tdErrEma = meta.tdErrEma
    return agent
  }

  getDiagnostics(): AgentDiagnostics {
    const params = this.net.paramCount
    return {
      paramCount: params,
      memoryBytes: params * 4 * (this.config.tc ? 3 : 1),
      // Effective α reflects annealing so the α chart shows the decay.
      learningRate: this.config.alpha * this.annealMult,
      meanAbsTdError: this.tdErrEma,
    }
  }
}
