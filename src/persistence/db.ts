// IndexedDB schema v2 (adds the highScoreGames store). Both the UI thread (library reads, fork/delete) and
// the training worker (checkpoints, buckets, session) open this database.
import { openDB, type DBSchema, type IDBPDatabase } from 'idb'
import type {
  CheckpointDoc,
  GameSummary,
  HighScoreGameRecord,
  ModelDoc,
  StatBucket,
  TrainingMode,
} from '../shared/types'

export interface SessionDoc {
  id: 'current'
  modelId: string | null
  wasTraining: boolean
  mode: TrainingMode
  movesPerSec: number
  updatedAt: number
}

export type StoredGame = GameSummary & { modelId: string }
export type StoredBucket = StatBucket & { modelId: string }
export type StoredHighScoreGame = HighScoreGameRecord & { modelId: string; id: string }

interface LabDB extends DBSchema {
  models: { key: string; value: ModelDoc }
  checkpoints: {
    key: string
    value: CheckpointDoc
    indexes: { 'by-model': string }
  }
  games: { key: [string, number]; value: StoredGame }
  statBuckets: { key: [string, number]; value: StoredBucket }
  highScoreGames: {
    key: string
    value: StoredHighScoreGame
    indexes: { 'by-model': string }
  }
  session: { key: string; value: SessionDoc }
}

export type LabDatabase = IDBPDatabase<LabDB>

let dbPromise: Promise<LabDatabase> | null = null

export function getDB(): Promise<LabDatabase> {
  if (!dbPromise) {
    dbPromise = openDB<LabDB>('ai2048lab', 2, {
      upgrade(db, oldVersion) {
        if (oldVersion < 1) {
          db.createObjectStore('models', { keyPath: 'id' })
          const cp = db.createObjectStore('checkpoints', { keyPath: 'id' })
          cp.createIndex('by-model', 'modelId')
          db.createObjectStore('games', { keyPath: ['modelId', 'game'] })
          db.createObjectStore('statBuckets', { keyPath: ['modelId', 'bucket'] })
          db.createObjectStore('session', { keyPath: 'id' })
        }
        if (!db.objectStoreNames.contains('highScoreGames')) {
          const hs = db.createObjectStore('highScoreGames', { keyPath: 'id' })
          hs.createIndex('by-model', 'modelId')
        }
      },
    })
  }
  return dbPromise
}
