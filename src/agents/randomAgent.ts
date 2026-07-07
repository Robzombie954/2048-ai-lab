// Uniform random legal play — the honest "no learning" baseline every
// learning curve is measured against (~1,090 average score).
import type { Board } from '../engine/board'
import { DIRS } from '../engine/rules'
import type { Rand } from '../engine/rng'
import type {
  Agent,
  AgentDiagnostics,
  AgentSnapshot,
  MoveEvaluation,
  Transition,
} from './types'

export class RandomAgent implements Agent {
  readonly kind = 'random' as const
  planningDepth = 1

  evaluateMoves(_state: Board, mask: number, rand: Rand): MoveEvaluation {
    const values: [number, number, number, number] = [-Infinity, -Infinity, -Infinity, -Infinity]
    const legal: number[] = []
    for (const dir of DIRS) {
      if (mask & (1 << dir)) {
        values[dir] = 0
        legal.push(dir)
      }
    }
    const chosen = legal[(rand() * legal.length) | 0]
    return { values, chosen: chosen as MoveEvaluation['chosen'], exploring: true }
  }

  observeTransition(_t: Transition): void {}
  onEpisodeEnd(_summary?: unknown): void {}

  serialize(): AgentSnapshot {
    return { metaJson: JSON.stringify({ version: 1, kind: 'random' }), buffers: [] }
  }

  getDiagnostics(): AgentDiagnostics {
    return { paramCount: 0, memoryBytes: 0, learningRate: 0, meanAbsTdError: 0 }
  }
}
