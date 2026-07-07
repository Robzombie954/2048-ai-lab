// Reads for charts (UI thread). All writes happen in the training worker.
import type { GameSummary, StatBucket } from '../shared/types'
import { normalizeBuckets } from '../stats/chartSeries'
import { getDB } from './db'

export async function getBuckets(modelId: string): Promise<StatBucket[]> {
  const db = await getDB()
  const rows = await db.getAll(
    'statBuckets',
    IDBKeyRange.bound([modelId, -Infinity], [modelId, Infinity]),
  )
  return normalizeBuckets(rows)
}

export async function getRecentGames(modelId: string, limit = 1000): Promise<GameSummary[]> {
  const db = await getDB()
  const rows = await db.getAll(
    'games',
    IDBKeyRange.bound([modelId, -Infinity], [modelId, Infinity]),
  )
  rows.sort((a, b) => a.game - b.game)
  return rows.slice(-limit)
}

