import { openDB } from 'idb'
import type { HighScoreGameRecord } from '../shared/types'

const DB_NAME = 'ai2048lab-high-score-archive'
const DB_VERSION = 1
const HANDLE_STORE = 'handles'
const PENDING_STORE = 'pending'
const ROOT_KEY = 'root'
export const HIGH_SCORE_ARCHIVE_DIR = '2048-ai-lab-high-score-games'

type DirectoryHandle = {
  name?: string
  queryPermission?: (opts: { mode: 'readwrite' }) => Promise<PermissionState>
  requestPermission?: (opts: { mode: 'readwrite' }) => Promise<PermissionState>
  getDirectoryHandle: (name: string, opts?: { create?: boolean }) => Promise<DirectoryHandle>
  getFileHandle: (name: string, opts?: { create?: boolean }) => Promise<FileHandle>
}

type FileHandle = {
  createWritable: () => Promise<{ write: (data: string) => Promise<void>; close: () => Promise<void> }>
}

export interface HighScoreArchiveStatus {
  supported: boolean
  configured: boolean
  ready: boolean
  label: string | null
  pending: number
  lastError?: string
}

export interface HighScoreArchiveResult extends HighScoreArchiveStatus {
  saved: boolean
  fileName?: string
}

function getPicker(): ((opts?: Record<string, unknown>) => Promise<DirectoryHandle>) | null {
  const picker = (globalThis as { showDirectoryPicker?: unknown }).showDirectoryPicker
  return typeof picker === 'function' ? (picker as (opts?: Record<string, unknown>) => Promise<DirectoryHandle>) : null
}

export function supportsHighScoreArchive(): boolean {
  return Boolean(getPicker()) && typeof indexedDB !== 'undefined'
}

async function archiveDB() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(HANDLE_STORE)) db.createObjectStore(HANDLE_STORE)
      if (!db.objectStoreNames.contains(PENDING_STORE)) db.createObjectStore(PENDING_STORE)
    },
  })
}

function recordKey(record: HighScoreGameRecord): string {
  return `${record.modelId}-${record.summary.game}-${record.summary.score}-${record.endedAt}`
}

function safePart(value: string): string {
  return value.replace(/[<>:"/\\|?*]+/g, '-').replace(/\s+/g, ' ').trim().slice(0, 80) || 'model'
}

export function highScoreFileName(record: HighScoreGameRecord): string {
  const when = new Date(record.endedAt).toISOString().replace(/[:.]/g, '-')
  const tile = record.summary.maxExp > 0 ? 1 << record.summary.maxExp : 0
  return `${safePart(record.modelName)}-game-${record.summary.game + 1}-score-${record.summary.score}-tile-${tile}-${when}.json`
}

function labelFor(root: DirectoryHandle | null): string | null {
  if (!root) return null
  const base = root.name?.trim() || 'selected folder'
  return `${base}\\${HIGH_SCORE_ARCHIVE_DIR}`
}

async function pendingCount(db?: any): Promise<number> {
  const d = db || await archiveDB()
  return d.count(PENDING_STORE)
}

async function getRoot(db?: any): Promise<DirectoryHandle | null> {
  const d = db || await archiveDB()
  return (await d.get(HANDLE_STORE, ROOT_KEY)) ?? null
}

async function hasWritePermission(root: DirectoryHandle): Promise<boolean> {
  if (!root.queryPermission) return true
  return (await root.queryPermission({ mode: 'readwrite' })) === 'granted'
}

async function requestWritePermission(root: DirectoryHandle): Promise<boolean> {
  if (await hasWritePermission(root)) return true
  if (!root.requestPermission) return false
  return (await root.requestPermission({ mode: 'readwrite' })) === 'granted'
}

async function archiveDir(root: DirectoryHandle): Promise<DirectoryHandle> {
  return root.getDirectoryHandle(HIGH_SCORE_ARCHIVE_DIR, { create: true })
}

async function writeRecord(root: DirectoryHandle, record: HighScoreGameRecord): Promise<string> {
  const dir = await archiveDir(root)
  const fileName = highScoreFileName(record)
  const file = await dir.getFileHandle(fileName, { create: true })
  const writable = await file.createWritable()
  await writable.write(JSON.stringify(record, null, 2))
  await writable.close()
  return fileName
}

export async function getHighScoreArchiveStatus(): Promise<HighScoreArchiveStatus> {
  if (!supportsHighScoreArchive()) {
    return {
      supported: false,
      configured: false,
      ready: false,
      label: null,
      pending: 0,
      lastError: 'This browser does not expose the directory picker.',
    }
  }
  const db = await archiveDB()
  const root = await getRoot(db)
  const pending = await pendingCount(db)
  if (!root) return { supported: true, configured: false, ready: false, label: null, pending }
  const ready = await hasWritePermission(root)
  return { supported: true, configured: true, ready, label: labelFor(root), pending }
}

export async function chooseHighScoreArchiveRoot(): Promise<HighScoreArchiveStatus> {
  const picker = getPicker()
  if (!picker) throw new Error('Directory picker is not available in this browser')
  const root = await picker({ id: 'ai2048-high-score-games', mode: 'readwrite' })
  if (!(await requestWritePermission(root))) throw new Error('Write permission was not granted')
  await archiveDir(root)
  const db = await archiveDB()
  await db.put(HANDLE_STORE, root, ROOT_KEY)
  await flushPendingHighScoreGames(root, db)
  return getHighScoreArchiveStatus()
}

export async function recordHighScoreGame(record: HighScoreGameRecord): Promise<HighScoreArchiveResult> {
  if (!supportsHighScoreArchive()) {
    return {
      supported: false,
      configured: false,
      ready: false,
      label: null,
      pending: 0,
      saved: false,
      lastError: 'This browser cannot save directly to a folder.',
    }
  }
  const db = await archiveDB()
  const key = recordKey(record)
  await db.put(PENDING_STORE, record, key)
  const root = await getRoot(db)
  if (!root) {
    return { supported: true, configured: false, ready: false, label: null, pending: await pendingCount(db), saved: false }
  }
  if (!(await hasWritePermission(root))) {
    return {
      supported: true,
      configured: true,
      ready: false,
      label: labelFor(root),
      pending: await pendingCount(db),
      saved: false,
      lastError: 'Folder permission needs to be renewed.',
    }
  }
  try {
    const fileName = await writeRecord(root, record)
    await db.delete(PENDING_STORE, key)
    return {
      supported: true,
      configured: true,
      ready: true,
      label: labelFor(root),
      pending: await pendingCount(db),
      saved: true,
      fileName,
    }
  } catch (err) {
    return {
      supported: true,
      configured: true,
      ready: false,
      label: labelFor(root),
      pending: await pendingCount(db),
      saved: false,
      lastError: err instanceof Error ? err.message : String(err),
    }
  }
}

async function flushPendingHighScoreGames(root: DirectoryHandle, db?: any): Promise<number> {
  const d = db || await archiveDB()
  const keys = await d.getAllKeys(PENDING_STORE)
  let saved = 0
  for (const key of keys) {
    const record = (await d.get(PENDING_STORE, key)) as HighScoreGameRecord | undefined
    if (!record) continue
    await writeRecord(root, record)
    await d.delete(PENDING_STORE, key)
    saved++
  }
  return saved
}
