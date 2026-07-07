// Domain types shared by agents, the training worker, persistence, and the UI.

export type TrainingMode = 'watch' | 'turbo' | 'grade'

export type NTuplePresetId = 'starter' | 'balanced' | 'expert'

/** Optional learning-rate annealing: the effective TD step decays as games
 * accumulate, quieting plateau noise once a model has mostly converged. */
export interface AnnealConfig {
  /** Games over which the effective learning-rate multiplier halves. */
  halfLifeGames: number
  /** Game index at which annealing began (multiplier = 1 here). */
  startGame: number
  /** Lower bound on the multiplier so learning never fully freezes. */
  floor: number
}

export interface NTupleConfig {
  kind: 'ntuple'
  preset: NTuplePresetId
  /** Total effective TD step per update, split across active weights. */
  alpha: number
  /** Temporal Coherence: per-weight self-tuning learning rates (3× memory). */
  tc: boolean
  /** Optimistic initialization: total initial V, drives early exploration. 0 = off. */
  optimisticInit: number
  /** 1 = pure learned policy; 2-3 = expectimax over spawns, learned V at leaves. */
  planningDepth: number
  /** κ multiplier for manual grades. */
  gradeStrength: number
  /** Learning-rate annealing schedule. Undefined = constant α (default). */
  anneal?: AnnealConfig
}

export interface RewardShaping {
  mode: 'scaled' | 'log'
  survivalBonus: boolean
  terminalPenalty: boolean
}

export type DQNAccelerator = 'auto' | 'webgpu' | 'cpu'

export interface DQNConfig {
  kind: 'dqn'
  accelerator?: DQNAccelerator
  hidden: number[]
  lr: number
  gamma: number
  epsStart: number
  epsEnd: number
  epsDecayMoves: number
  replaySize: number
  batchSize: number
  /** Learn step every N moves. */
  trainFreq: number
  /** Sync target net every N learn steps. */
  targetSync: number
  shaping: RewardShaping
  gradeStrength: number
}

export type ModelConfig = NTupleConfig | DQNConfig

export interface Milestone {
  exp: number // tile exponent reached for the first time (7 = 128 … 15 = 32768)
  game: number
  totalMoves: number
  wallMs: number
  at: number // epoch ms
}



export interface GameSummary {
  game: number
  score: number
  moves: number
  maxExp: number
  seed: number
}

export interface HighScoreGameRecord {
  version: 1
  endedAt: number
  modelId: string
  modelName: string
  modelKind: 'ntuple' | 'dqn'
  config: ModelConfig
  summary: GameSummary
  previousBestScore: number
  previousBestExp: number
  finalBoard: number[]
  /** Direction ids: 0=up, 1=right, 2=down, 3=left. Seed + actions replay the game. */
  actions: number[]
  actionLegend: readonly ['up', 'right', 'down', 'left']
  totalMovesAt: number
  trainMsAt: number
  /** Internal storage key (not serialized to archive files). */
  id?: string
}
export interface TrajectoryPoint {
  move: number
  score: number
}

export const BUCKET_SIZE = 100

/** Permanent per-100-game aggregate — the unit of long-run chart storage. */
export interface StatBucket {
  bucket: number
  n: number
  scoreSum: number
  scoreSqSum: number
  scoreMin: number
  scoreMax: number
  movesSum: number
  /** maxTileHist[e] = games in this bucket whose best tile had exponent e. */
  maxTileHist: number[]
  tdErrAvg: number
  epsAvg: number
  lrAvg: number
  wallMs: number
  endTs: number
}

export interface LiveStats {
  games: number
  totalMoves: number
  trainMs: number
  movesPerSec: number
  /** Games completed per second (EMA). */
  gamesPerSec?: number
  currentScore: number
  currentMaxExp: number
  bestScore: number
  bestExp: number
  recentAvg: number
  paramCount: number
  memoryBytes: number
  learningRate: number
  epsilon?: number
  meanAbsTdError: number
  replayFill?: number
  accelerator?: string
  acceleratorDetail?: string
  gpuQueue?: number
}

export interface ModelDoc {
  id: string
  name: string
  kind: 'ntuple' | 'dqn'
  config: ModelConfig
  createdAt: number
  updatedAt: number
  games: number
  moves: number
  trainMs: number
  bestScore: number
  bestExp: number
  bestRollingAvg: number
  milestones: Milestone[]
  latestCheckpointId: string | null
  bestCheckpointId: string | null
  forkedFrom?: string
}

export interface CheckpointDoc {
  id: string
  modelId: string
  createdAt: number
  gamesAt: number
  movesAt: number
  trainMsAt: number
  rollingAvg: number
  reason: 'auto' | 'manual' | 'stop'
  metaJson: string
  buffers: ArrayBuffer[]
  sizeBytes: number
}




