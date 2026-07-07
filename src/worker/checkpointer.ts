// All persistence the training worker performs: checkpoints (with
// retention), game summaries, stat buckets, session doc, and the
// consistency rollback that keeps stats aligned with restored weights.
import type { AgentSnapshot } from '../agents/types'
import { BUCKET_SIZE, type CheckpointDoc, type HighScoreGameRecord, type ModelDoc, type StatBucket, type GameSummary, type TrainingMode } from '../shared/types'
import { getDB, type StoredHighScoreGame } from '../persistence/db'
import { uuid } from '../lib/uuid'

const KEEP_LATEST = 2
const GAMES_RETAINED = 1000

export interface Counters {
  games: number
  moves: number
  trainMs: number
  rollingAvg: number
}

export async function saveCheckpoint(
  doc: ModelDoc,
  snap: AgentSnapshot,
  counters: Counters,
  reason: 'auto' | 'manual' | 'stop',
): Promise<CheckpointDoc> {
  const db = await getDB()
  let sizeBytes = snap.metaJson.length
  for (const b of snap.buffers) sizeBytes += b.byteLength
  const cp: CheckpointDoc = {
    id: uuid(),
    modelId: doc.id,
    createdAt: Date.now(),
    gamesAt: counters.games,
    movesAt: counters.moves,
    trainMsAt: counters.trainMs,
    rollingAvg: counters.rollingAvg,
    reason,
    metaJson: snap.metaJson,
    buffers: snap.buffers,
    sizeBytes,
  }
  await db.put('checkpoints', cp)

  doc.games = counters.games
  doc.moves = counters.moves
  doc.trainMs = counters.trainMs
  doc.updatedAt = Date.now()
  doc.latestCheckpointId = cp.id
  if (counters.rollingAvg >= doc.bestRollingAvg) {
    doc.bestRollingAvg = counters.rollingAvg
    doc.bestCheckpointId = cp.id
  }
  await db.put('models', doc)

  // Retention: newest KEEP_LATEST plus the best-rolling-avg checkpoint.
  const all = await db.getAllFromIndex('checkpoints', 'by-model', doc.id)
  all.sort((a, b) => b.createdAt - a.createdAt)
  const keep = new Set<string>(all.slice(0, KEEP_LATEST).map((c) => c.id))
  if (doc.bestCheckpointId) keep.add(doc.bestCheckpointId)
  for (const old of all) {
    if (!keep.has(old.id)) await db.delete('checkpoints', old.id)
  }

  // Bound the raw game-summary window.
  if (counters.games > GAMES_RETAINED) {
    await db.delete(
      'games',
      IDBKeyRange.bound([doc.id, -Infinity], [doc.id, counters.games - GAMES_RETAINED], false, true),
    )
  }
  return cp
}

export async function persistGame(modelId: string, summary: GameSummary): Promise<void> {
  const db = await getDB()
  await db.put('games', { ...summary, modelId })
}

export async function persistBucket(modelId: string, bucket: StatBucket): Promise<void> {
  const db = await getDB()
  await db.put('statBuckets', { ...bucket, modelId })
}

function makeHighScoreId(record: HighScoreGameRecord): string {
  return `${record.modelId}-${record.summary.game}-${record.endedAt}`
}

export async function persistHighScoreGame(record: HighScoreGameRecord): Promise<void> {
  const db = await getDB()
  const withId: StoredHighScoreGame = {
    ...(record as any),
    id: makeHighScoreId(record),
    modelId: record.modelId,
  } as StoredHighScoreGame
  await db.put('highScoreGames', withId)
}

export async function getHighScoreGames(modelId: string): Promise<HighScoreGameRecord[]> {
  const db = await getDB()
  const rows = await db.getAllFromIndex('highScoreGames', 'by-model', modelId)
  // Sort by game number asc (chronological improvement order)
  rows.sort((a, b) => a.summary.game - b.summary.game)
  // Strip the extra id field for callers
  return rows.map((r) => {
    const { id: _id, ...clean } = r as any
    return clean as HighScoreGameRecord
  })
}

/**
 * After restoring a checkpoint at `games` completed games, remove any stats
 * recorded past that point so charts and weights tell the same story.
 */
export async function rollbackBeyond(modelId: string, games: number): Promise<void> {
  const db = await getDB()
  // A bucket is valid only if fully contained in the completed range.
  const lastValidBucket = Math.floor(games / BUCKET_SIZE) - 1
  await db.delete(
    'statBuckets',
    IDBKeyRange.bound([modelId, lastValidBucket], [modelId, Infinity], true, false),
  )
  await db.delete('games', IDBKeyRange.bound([modelId, games], [modelId, Infinity], false, false))
}

export async function loadResumeData(
  modelId: string,
  games: number,
): Promise<{ recentScores: number[]; partialBucketGames: GameSummary[] }> {
  const db = await getDB()
  const rows = await db.getAll(
    'games',
    IDBKeyRange.bound([modelId, -Infinity], [modelId, Infinity]),
  )
  rows.sort((a, b) => a.game - b.game)
  const bucketStart = Math.floor(games / BUCKET_SIZE) * BUCKET_SIZE
  return {
    recentScores: rows.slice(-100).map((r) => r.score),
    partialBucketGames: rows.filter((r) => r.game >= bucketStart && r.game < games),
  }
}

export async function updateSession(
  modelId: string | null,
  wasTraining: boolean,
  mode: TrainingMode,
  movesPerSec: number,
): Promise<void> {
  const db = await getDB()
  await db.put('session', {
    id: 'current',
    modelId,
    wasTraining,
    mode,
    movesPerSec,
    updatedAt: Date.now(),
  })
}
