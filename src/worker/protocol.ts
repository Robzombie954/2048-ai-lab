// The UI ↔ trainer-worker contract. Boards travel as plain number[16]
// exponent arrays; weight exports travel as transferred ArrayBuffers.
import type { AgentDiagnostics, Grade } from '../agents/types'
import type { Dir } from '../engine/rules'
import type {
  GameSummary,
  HighScoreGameRecord,
  LiveStats,
  Milestone,
  ModelConfig,
  ModelDoc,
  StatBucket,
  TrainingMode,
} from '../shared/types'

export type WorkerCommand =
  | { type: 'createModel'; name: string; config: ModelConfig }
  | { type: 'loadModel'; modelId: string }
  | { type: 'start'; mode: TrainingMode; movesPerSec: number }
  | { type: 'stop' }
  | { type: 'setMode'; mode: TrainingMode }
  | { type: 'setSpeed'; movesPerSec: number }
  | { type: 'setPlanningDepth'; depth: number }
  | { type: 'setAnnealing'; enabled: boolean; halfLifeGames?: number }
  | { type: 'grade'; grade: Grade }
  | { type: 'newGame' }
  | { type: 'playOneGame'; movesPerSec: number }
  | { type: 'trainForGames'; count: number }
  | { type: 'saveCheckpoint'; reason: 'manual' | 'stop' }
  | { type: 'exportModel' }
  | { type: 'dispose' }

export interface MoveEventPayload {
  gameIndex: number
  moveIndex: number
  action: Dir
  /** Learned per-direction values at decision time (null = illegal). */
  values: (number | null)[]
  exploring: boolean
  spawn: { idx: number; exp: number } | null
  postBoard: number[]
  score: number
  terminal: boolean
  /** Grade mode: the worker is paused waiting for a grade on this move. */
  awaitingGrade: boolean
}

export type WorkerEvent =
  | { type: 'ready' }
  | {
      type: 'modelLoaded'
      doc: ModelDoc
      diagnostics: AgentDiagnostics
      board: number[]
      score: number
    }
  | { type: 'modelDisposed' }
  | { type: 'trainingState'; running: boolean; mode: TrainingMode; movesPerSec: number; oneGameActive: boolean }
  | { type: 'boardSync'; gameIndex: number; moveIndex: number; board: number[]; score: number; terminal: boolean }
  | { type: 'moveEvent'; move: MoveEventPayload }
  | {
      type: 'snapshot'
      board: number[]
      score: number
      gameIndex: number
      values: (number | null)[]
      lastAction: Dir | null
    }
  | {
      type: 'stats'
      live: LiveStats
      partialBucket: StatBucket | null
      completedBucket?: StatBucket
    }
  | { type: 'episodeEnd'; summary: GameSummary }
  | { type: 'highScoreGame'; record: HighScoreGameRecord }
  | { type: 'milestone'; milestone: Milestone }
  | {
      type: 'checkpointSaved'
      checkpointId: string
      sizeBytes: number
      reason: string
      gamesAt: number
    }
  | { type: 'exportReady'; fileName: string; data: ArrayBuffer }
  | { type: 'modelsChanged' }
  | { type: 'error'; message: string }


