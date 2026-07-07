// Model library CRUD used by the UI thread. The training worker is the only
// writer for the ACTIVE model's doc/checkpoints; the UI must dispose the
// active model before deleting it.
import type { ModelConfig, ModelDoc } from '../shared/types'
import { uuid } from '../lib/uuid'
import { getDB } from './db'
import type { ParsedExport } from './exportImport'

export function newModelDoc(name: string, config: ModelConfig): ModelDoc {
  const now = Date.now()
  return {
    id: uuid(),
    name,
    kind: config.kind,
    config,
    createdAt: now,
    updatedAt: now,
    games: 0,
    moves: 0,
    trainMs: 0,
    bestScore: 0,
    bestExp: 0,
    bestRollingAvg: 0,
    milestones: [],
    latestCheckpointId: null,
    bestCheckpointId: null,
  }
}

export async function listModels(): Promise<ModelDoc[]> {
  const db = await getDB()
  const docs = await db.getAll('models')
  return docs.sort((a, b) => b.updatedAt - a.updatedAt)
}

export async function getModel(id: string): Promise<ModelDoc | undefined> {
  const db = await getDB()
  return db.get('models', id)
}

export async function renameModel(id: string, name: string): Promise<void> {
  const db = await getDB()
  const doc = await db.get('models', id)
  if (!doc) return
  doc.name = name
  doc.updatedAt = Date.now()
  await db.put('models', doc)
}

/** Cascade delete: doc, checkpoints, game summaries, stat buckets, high score records. */
export async function deleteModel(id: string): Promise<void> {
  const db = await getDB()
  const checkpoints = await db.getAllKeysFromIndex('checkpoints', 'by-model', id)
  const tx = db.transaction(['models', 'checkpoints', 'games', 'statBuckets', 'highScoreGames'], 'readwrite')
  await tx.objectStore('models').delete(id)
  for (const key of checkpoints) await tx.objectStore('checkpoints').delete(key)
  await tx
    .objectStore('games')
    .delete(IDBKeyRange.bound([id, -Infinity], [id, Infinity]))
  await tx
    .objectStore('statBuckets')
    .delete(IDBKeyRange.bound([id, -Infinity], [id, Infinity]))
  // highScoreGames keyed by string id; delete via index
  const hsKeys = await db.getAllKeysFromIndex('highScoreGames', 'by-model', id)
  for (const k of hsKeys) tx.objectStore('highScoreGames').delete(k)
  await tx.done
}

/** Recreate a model (weights, stats history, milestones) from a .2048model file. */
export async function importModelFromExport(parsed: ParsedExport): Promise<ModelDoc> {
  const db = await getDB()
  const h = parsed.header
  const doc: ModelDoc = {
    ...h.model,
    id: uuid(),
    latestCheckpointId: null,
    bestCheckpointId: null,
    updatedAt: Date.now(),
  }
  const cpId = uuid()
  doc.latestCheckpointId = cpId
  let sizeBytes = h.checkpoint.metaJson.length
  for (const b of parsed.buffers) sizeBytes += b.byteLength
  await db.put('checkpoints', {
    id: cpId,
    modelId: doc.id,
    createdAt: Date.now(),
    gamesAt: h.checkpoint.gamesAt,
    movesAt: h.checkpoint.movesAt,
    trainMsAt: h.checkpoint.trainMsAt,
    rollingAvg: h.checkpoint.rollingAvg,
    reason: 'manual',
    metaJson: h.checkpoint.metaJson,
    buffers: parsed.buffers,
    sizeBytes,
  })
  for (const bucket of h.buckets) {
    await db.put('statBuckets', { ...bucket, modelId: doc.id })
  }
  await db.put('models', doc)
  return doc
}

/**
 * Fork: new model with the parent's config and current weights, stats reset
 * to zero. The parent is untouched.
 */
export async function forkModel(parentId: string, name: string): Promise<ModelDoc | null> {
  const db = await getDB()
  const parent = await db.get('models', parentId)
  if (!parent) return null
  const source = parent.latestCheckpointId
    ? await db.get('checkpoints', parent.latestCheckpointId)
    : undefined
  const doc = newModelDoc(name, parent.config)
  doc.forkedFrom = parent.name
  if (source) {
    const checkpointId = uuid()
    doc.latestCheckpointId = checkpointId
    await db.put('checkpoints', {
      id: checkpointId,
      modelId: doc.id,
      createdAt: Date.now(),
      gamesAt: 0,
      movesAt: 0,
      trainMsAt: 0,
      rollingAvg: 0,
      reason: 'manual',
      metaJson: source.metaJson,
      buffers: source.buffers.map((b) => b.slice(0)),
      sizeBytes: source.sizeBytes,
    })
  }
  await db.put('models', doc)
  return doc
}
