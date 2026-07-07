// .2048model binary container:
//   magic "2048AILAB\x01" (10 bytes) · u32le header length · UTF-8 JSON
//   header · 8-byte-aligned raw weight buffers.
// Round-trips byte-exactly (tested).
import type { CheckpointDoc, ModelDoc, StatBucket } from '../shared/types'

const MAGIC = '2048AILAB\x01'

export interface ExportHeader {
  formatVersion: 1
  model: Omit<ModelDoc, 'id' | 'latestCheckpointId' | 'bestCheckpointId'>
  checkpoint: {
    gamesAt: number
    movesAt: number
    trainMsAt: number
    rollingAvg: number
    metaJson: string
    bufferLengths: number[]
  }
  buckets: StatBucket[]
}

export interface ParsedExport {
  header: ExportHeader
  buffers: ArrayBuffer[]
}

const align8 = (n: number) => (n + 7) & ~7

export function buildExport(
  doc: ModelDoc,
  checkpoint: Pick<CheckpointDoc, 'gamesAt' | 'movesAt' | 'trainMsAt' | 'rollingAvg' | 'metaJson' | 'buffers'>,
  buckets: StatBucket[],
): ArrayBuffer {
  const { id: _id, latestCheckpointId: _l, bestCheckpointId: _b, ...model } = doc
  const header: ExportHeader = {
    formatVersion: 1,
    model,
    checkpoint: {
      gamesAt: checkpoint.gamesAt,
      movesAt: checkpoint.movesAt,
      trainMsAt: checkpoint.trainMsAt,
      rollingAvg: checkpoint.rollingAvg,
      metaJson: checkpoint.metaJson,
      bufferLengths: checkpoint.buffers.map((b) => b.byteLength),
    },
    buckets,
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header))
  let offset = align8(MAGIC.length + 4 + headerBytes.length)
  const dataStart = offset
  for (const b of checkpoint.buffers) offset = align8(offset + b.byteLength)
  const out = new ArrayBuffer(offset)
  const bytes = new Uint8Array(out)
  const view = new DataView(out)
  for (let i = 0; i < MAGIC.length; i++) bytes[i] = MAGIC.charCodeAt(i)
  view.setUint32(MAGIC.length, headerBytes.length, true)
  bytes.set(headerBytes, MAGIC.length + 4)
  let pos = dataStart
  for (const b of checkpoint.buffers) {
    bytes.set(new Uint8Array(b), pos)
    pos = align8(pos + b.byteLength)
  }
  return out
}

export function parseExport(data: ArrayBuffer): ParsedExport {
  const bytes = new Uint8Array(data)
  if (bytes.length < MAGIC.length + 4) throw new Error('file too small to be a .2048model')
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC.charCodeAt(i)) throw new Error('not a .2048model file (bad magic)')
  }
  const view = new DataView(data)
  const headerLen = view.getUint32(MAGIC.length, true)
  const headerStart = MAGIC.length + 4
  if (headerStart + headerLen > bytes.length) throw new Error('corrupt header length')
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen)),
  ) as ExportHeader
  if (header.formatVersion !== 1) {
    throw new Error(`unsupported format version ${header.formatVersion}`)
  }
  const buffers: ArrayBuffer[] = []
  let pos = align8(headerStart + headerLen)
  for (const len of header.checkpoint.bufferLengths) {
    if (pos + len > bytes.length) throw new Error('corrupt buffer table')
    buffers.push(data.slice(pos, pos + len))
    pos = align8(pos + len)
  }
  return { header, buffers }
}
