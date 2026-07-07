import { create } from 'zustand'
import type { AgentDiagnostics } from '../agents/types'
import type { Dir } from '../engine/rules'
import type {
  GameSummary,
  HighScoreGameRecord,
  LiveStats,
  Milestone,
  ModelDoc,
  NTuplePresetId,
  StatBucket,
  TrainingMode,
  TrajectoryPoint,
} from '../shared/types'

/** Pre-seeds the New-Model wizard when "leveling up" from an existing model. */
export interface WizardSeed {
  preset: NTuplePresetId
  fromName: string
  autostart: boolean
}

export interface RenderTile {
  id: number
  idx: number
  exp: number
  /** Merge victim kept around one beat so it can slide under the survivor. */
  ghost?: boolean
  justMerged?: boolean
  justSpawned?: boolean
}

export interface ResumeHint {
  modelId: string
  name: string
  mode: TrainingMode
  movesPerSec: number
  ageMs: number
}

interface LabState {
  ready: boolean
  models: ModelDoc[]
  activeDoc: ModelDoc | null
  diagnostics: AgentDiagnostics | null

  running: boolean
  mode: TrainingMode
  movesPerSec: number
  planningDepth: number

  live: LiveStats | null
  buckets: StatBucket[]
  partialBucket: StatBucket | null
  milestones: Milestone[]

  tiles: RenderTile[]
  instantBoard: boolean
  score: number
  gameIndex: number
  lastValues: (number | null)[]
  lastAction: Dir | null
  exploring: boolean
  awaitingGrade: boolean
  oneGameActive: boolean
  gameOverSummary: GameSummary | null
  trajectory: TrajectoryPoint[]

  lastCheckpointAt: number | null
  lastCheckpointBytes: number
  wizardOpen: boolean
  wizardSeed: WizardSeed | null
  /** When set, the next model that finishes loading auto-starts training. */
  autostartOnLoad: boolean
  resumeHint: ResumeHint | null
  toast: string | null
  errorMsg: string | null

  // High score history for the active model (record-breaking games only)
  highScoreGames: HighScoreGameRecord[]

  // On-demand replay of a saved high-score game recipe
  replay: {
    record: HighScoreGameRecord
    index: number // how many actions applied so far
    playing: boolean
    speed: number // actions per sec for animation
    // Playlist support for "play all"
    playlistTotal?: number
    playlistCurrent?: number
  } | null

  // For UI indication of batch target
  targetGameCount: number | null
}

export const useLabStore = create<LabState>(() => ({
  ready: false,
  models: [],
  activeDoc: null,
  diagnostics: null,

  running: false,
  mode: 'turbo',
  movesPerSec: 6,
  planningDepth: 1,

  live: null,
  buckets: [],
  partialBucket: null,
  milestones: [],

  tiles: [],
  instantBoard: true,
  score: 0,
  gameIndex: 0,
  lastValues: [null, null, null, null],
  lastAction: null,
  exploring: false,
  awaitingGrade: false,
  oneGameActive: false,
  gameOverSummary: null,
  trajectory: [{ move: 0, score: 0 }],

  lastCheckpointAt: null,
  lastCheckpointBytes: 0,
  wizardOpen: false,
  wizardSeed: null,
  autostartOnLoad: false,
  resumeHint: null,
  toast: null,
  errorMsg: null,

  highScoreGames: [],
  replay: null,
  targetGameCount: null,
}))

export const setLab = useLabStore.setState
export const getLab = useLabStore.getState

