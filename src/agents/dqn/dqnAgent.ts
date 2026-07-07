// DQN: epsilon-greedy Q-learning over a hand-rolled MLP, experience replay,
// target network, Huber loss, illegal actions masked at both selection and
// bootstrap. The honest, slow, "real neural network" path.
import type { Board } from '../../engine/board'
import { DIRS } from '../../engine/rules'
import { legalMask } from '../../engine/fastEngine'
import { splitmix32, randomSeed, type Rand } from '../../engine/rng'
import type { DQNConfig } from '../../shared/types'
import type {
  Agent,
  AgentDiagnostics,
  AgentSnapshot,
  MoveEvaluation,
  Transition,
} from '../types'
import { DQN_INPUT, encodeBoard } from './encoding'
import { MLP } from './mlp'
import { ReplayBuffer } from './replay'

export class DQNAgent implements Agent {
  readonly kind = 'dqn' as const
  planningDepth = 1 // DQN has no planning mode; kept for the interface

  private readonly config: DQNConfig
  private readonly online: MLP
  private readonly target: MLP
  private readonly replay: ReplayBuffer
  private readonly acts: Float32Array[]
  private readonly targetActs: Float32Array[]
  private readonly dOut: Float32Array
  private readonly sampleBatch: Uint32Array
  private readonly scratchState: Uint8Array
  private readonly scratchNext: Uint8Array
  private readonly warmup: number
  private readonly sampleRand: Rand

  private moveCount = 0
  private learnSteps = 0
  private meanAbsShaped = 0
  private tdErrEma = 0

  constructor(config: DQNConfig, seed: number = randomSeed()) {
    this.config = config
    const sizes = [DQN_INPUT, ...config.hidden, 4]
    const initRand = splitmix32(seed)
    this.online = new MLP(sizes, initRand)
    this.target = new MLP(sizes)
    this.target.copyWeightsFrom(this.online)
    this.replay = new ReplayBuffer(config.replaySize)
    this.acts = this.online.makeActs()
    this.targetActs = this.target.makeActs()
    this.dOut = new Float32Array(4)
    this.sampleBatch = new Uint32Array(config.batchSize)
    this.scratchState = new Uint8Array(16)
    this.scratchNext = new Uint8Array(16)
    this.warmup = Math.min(1000, Math.max(config.batchSize * 4, config.replaySize / 10))
    this.sampleRand = splitmix32(seed ^ 0x7ab1e5)
  }

  epsilon(): number {
    const { epsStart, epsEnd, epsDecayMoves } = this.config
    return epsEnd + (epsStart - epsEnd) * Math.exp(-this.moveCount / epsDecayMoves)
  }

  evaluateMoves(state: Board, mask: number, rand: Rand): MoveEvaluation {
    encodeBoard(state, this.acts[0])
    const q = this.online.forward(this.acts)
    const values: [number, number, number, number] = [-Infinity, -Infinity, -Infinity, -Infinity]
    const legal: number[] = []
    let best = -Infinity
    let greedy = 0
    for (const dir of DIRS) {
      if (!(mask & (1 << dir))) continue
      values[dir] = q[dir]
      legal.push(dir)
      if (q[dir] > best) {
        best = q[dir]
        greedy = dir
      }
    }
    if (rand() < this.epsilon()) {
      const chosen = legal[(rand() * legal.length) | 0]
      return { values, chosen: chosen as MoveEvaluation['chosen'], exploring: true }
    }
    return { values, chosen: greedy as MoveEvaluation['chosen'], exploring: false }
  }

  private shapeReward(raw: number, terminal: boolean): number {
    const s = this.config.shaping
    let r = s.mode === 'log' ? Math.log2(1 + raw) / 10 : raw / 1024
    if (s.survivalBonus) r += 0.01
    if (s.terminalPenalty && terminal) r -= 1
    return r
  }

  observeTransition(t: Transition): void {
    let r = this.shapeReward(t.reward, t.terminal)
    if (t.grade) {
      r += this.config.gradeStrength * t.grade * Math.max(this.meanAbsShaped, 0.05)
    }
    this.meanAbsShaped += 0.001 * (Math.abs(r) - this.meanAbsShaped)
    const idx = this.replay.push(t.state, t.action, r, t.next, t.terminal, !!t.grade)
    this.moveCount++
    // A grade teaches immediately, then keeps teaching via boosted sampling.
    if (t.grade) {
      this.sampleBatch[0] = idx
      this.learnOn(this.sampleBatch, 1)
    }
    if (this.moveCount % this.config.trainFreq === 0 && this.replay.size >= this.warmup) {
      this.replay.fillSample(this.sampleRand, this.sampleBatch, this.config.batchSize)
      this.learnOn(this.sampleBatch, this.config.batchSize)
    }
  }

  private learnOn(indices: Uint32Array, count: number): void {
    this.online.zeroGrads()
    for (let bi = 0; bi < count; bi++) {
      const i = indices[bi]
      let y = this.replay.rewardAt(i)
      if (!this.replay.terminalAt(i)) {
        this.replay.copyNextInto(i, this.scratchNext)
        const mask = legalMask(this.scratchNext)
        if (mask !== 0) {
          encodeBoard(this.scratchNext, this.targetActs[0])
          const qNext = this.target.forward(this.targetActs)
          let best = -Infinity
          for (const dir of DIRS) {
            if (mask & (1 << dir) && qNext[dir] > best) best = qNext[dir]
          }
          y += this.config.gamma * best
        }
      }
      this.replay.copyStateInto(i, this.scratchState)
      encodeBoard(this.scratchState, this.acts[0])
      const q = this.online.forward(this.acts)
      const a = this.replay.actionAt(i)
      const err = q[a] - y
      this.tdErrEma += 0.001 * (Math.abs(err) - this.tdErrEma)
      this.dOut.fill(0)
      this.dOut[a] = err > 1 ? 1 : err < -1 ? -1 : err // Huber gradient
      this.online.backward(this.acts, this.dOut)
    }
    this.online.adamStep(this.config.lr, 1 / count)
    this.learnSteps++
    if (this.learnSteps % this.config.targetSync === 0) {
      this.target.copyWeightsFrom(this.online)
    }
  }

  onEpisodeEnd(_summary?: unknown): void {}

  serialize(): AgentSnapshot {
    return {
      metaJson: JSON.stringify({
        version: 1,
        config: this.config,
        moveCount: this.moveCount,
        learnSteps: this.learnSteps,
        meanAbsShaped: this.meanAbsShaped,
        tdErrEma: this.tdErrEma,
        adamT: this.online.adamT,
      }),
      // Replay is deliberately not persisted (tens of MB); it refills during
      // play and learning resumes after the warmup window.
      buffers: this.online.serializeBuffers(),
    }
  }

  static restore(metaJson: string, buffers: ArrayBuffer[]): DQNAgent {
    const meta = JSON.parse(metaJson) as {
      config: DQNConfig
      moveCount: number
      learnSteps: number
      meanAbsShaped: number
      tdErrEma: number
      adamT: number
    }
    const agent = new DQNAgent(meta.config)
    agent.online.restoreBuffers(buffers)
    agent.online.adamT = meta.adamT
    agent.target.copyWeightsFrom(agent.online)
    agent.moveCount = meta.moveCount
    agent.learnSteps = meta.learnSteps
    agent.meanAbsShaped = meta.meanAbsShaped
    agent.tdErrEma = meta.tdErrEma
    return agent
  }

  getDiagnostics(): AgentDiagnostics {
    return {
      paramCount: this.online.paramCount,
      memoryBytes: this.online.paramCount * 4 * 3 + this.replay.cap * 40,
      learningRate: this.config.lr,
      epsilon: this.epsilon(),
      meanAbsTdError: this.tdErrEma,
      replayFill: this.replay.size / this.replay.cap,
    }
  }
}

