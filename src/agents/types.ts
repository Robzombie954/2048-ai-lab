// The common Agent contract. The training loop, the grading flow, the
// DirectionViz, and checkpointing all speak only this interface — no agent
// implementation may contain hand-written board-quality heuristics; move
// choices must derive solely from learned parameters plus engine legality.
import type { Board } from '../engine/board'
import type { Dir } from '../engine/rules'
import type { Rand } from '../engine/rng'
import type { AnnealConfig } from '../shared/types'

export type AgentKind = 'ntuple' | 'dqn' | 'random'
export type Grade = -1 | 0 | 1

export interface MoveEvaluation {
  /** Learned per-direction value; -Infinity for illegal directions. */
  values: [number, number, number, number]
  chosen: Dir
  /** True when the pick was an exploration move (e.g. DQN epsilon). */
  exploring: boolean
}

export interface Transition {
  /** Pre-move state s (after the previous spawn). */
  state: Board
  action: Dir
  /** Raw merge score gained by the move. */
  reward: number
  /** Post-slide, pre-spawn afterstate s'. */
  afterstate: Board
  /** Post-spawn state s''. */
  next: Board
  /** True when s'' has no legal moves. */
  terminal: boolean
  /** Manual grading signal, when in grade mode. */
  grade?: Grade
}

export interface EpisodeSummary {
  score: number
  moves: number
  maxExp: number
}

export interface AgentDiagnostics {
  paramCount: number
  memoryBytes: number
  learningRate: number
  epsilon?: number
  meanAbsTdError: number
  replayFill?: number
}

export interface AgentSnapshot {
  metaJson: string
  buffers: ArrayBuffer[]
}

export interface Agent {
  readonly kind: AgentKind
  planningDepth: number
  /** Boards passed in are borrowed — copy anything retained past the call. */
  evaluateMoves(state: Board, mask: number, rand: Rand): MoveEvaluation
  observeTransition(t: Transition): void
  onEpisodeEnd(summary: EpisodeSummary): void
  serialize(): AgentSnapshot
  getDiagnostics(): AgentDiagnostics
  /** Optional per-game schedule hook (e.g. learning-rate annealing). */
  refreshSchedule?(gamesSeen: number): void
  /** Optional runtime enable/disable of a learning-rate anneal schedule. */
  setAnneal?(anneal: AnnealConfig | null, gamesSeen: number): void
}
