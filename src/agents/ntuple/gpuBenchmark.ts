import type { NTuplePresetId } from '../../shared/types'
import { splitmix32 } from '../../engine/rng'
import { NTUPLE_PRESETS } from './patterns'
import { NTupleNetwork } from './network'

type BenchmarkStatus = 'ok' | 'unavailable' | 'error'

export interface NTupleGpuBenchmarkResult {
  status: BenchmarkStatus
  preset: NTuplePresetId
  boards: number
  cpuMs: number
  gpuMs: number
  gpuComputeMs: number
  gpuReadbackMs: number
  cpuBoardsPerSec: number
  gpuBoardsPerSec: number
  speedup: number
  detail: string
  checksumDiff: number
}

function makeBoards(count: number): Uint8Array {
  const rand = splitmix32(0x2048b33f)
  const boards = new Uint8Array(count * 16)
  for (let b = 0; b < count; b++) {
    for (let i = 0; i < 16; i++) {
      const r = rand()
      boards[b * 16 + i] = r < 0.22 ? 0 : Math.min(15, 1 + ((rand() * 11) | 0))
    }
  }
  return boards
}

function fillSyntheticWeights(net: NTupleNetwork): void {
  for (let p = 0; p < net.tables.length; p++) {
    const table = net.tables[p]
    for (let i = 0; i < table.length; i++) {
      table[i] = ((i * 1664525 + p * 1013904223) & 0xffff) / 0xffff - 0.5
    }
  }
}

function cpuBenchmark(net: NTupleNetwork, boards: Uint8Array, count: number): { ms: number; checksum: number } {
  let checksum = 0
  const t0 = performance.now()
  for (let b = 0; b < count; b++) {
    const boardBase = b * 16
    let value = 0
    for (let vi = 0; vi < net.totalViews; vi++) {
      const p = net.viewPattern[vi]
      const len = net.tupleLens[p]
      const off = net.viewOffsets[vi]
      let idx = 0
      for (let j = 0; j < len; j++) idx = (idx << 4) | boards[boardBase + net.viewCells[off + j]]
      value += net.tables[p][idx]
    }
    checksum += value
  }
  return { ms: performance.now() - t0, checksum }
}

function flattenTables(net: NTupleNetwork): { weights: Float32Array; offsets: Uint32Array } {
  const offsets = new Uint32Array(net.tables.length)
  let total = 0
  for (let i = 0; i < net.tables.length; i++) {
    offsets[i] = total
    total += net.tables[i].length
  }
  const weights = new Float32Array(total)
  let at = 0
  for (const table of net.tables) {
    weights.set(table, at)
    at += table.length
  }
  return { weights, offsets }
}

const wgsl = `
struct Dims { boards:u32, views:u32, _pad0:u32, _pad1:u32 }
@group(0) @binding(0) var<storage, read> boards: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> tableOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> tupleLens: array<u32>;
@group(0) @binding(4) var<storage, read> viewPattern: array<u32>;
@group(0) @binding(5) var<storage, read> viewOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> viewCells: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn valueKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= dims.boards) { return; }
  var sum = 0.0;
  let boardBase = b * 16u;
  for (var v:u32 = 0u; v < dims.views; v = v + 1u) {
    let p = viewPattern[v];
    let len = tupleLens[p];
    let off = viewOffsets[v];
    var idx = 0u;
    for (var j:u32 = 0u; j < len; j = j + 1u) {
      idx = (idx << 4u) | boards[boardBase + viewCells[off + j]];
    }
    sum = sum + weights[tableOffsets[p] + idx];
  }
  out[b] = sum;
}
`

function createBuffer(device: any, data: ArrayBufferView, usage: number): any {
  const GPUBufferUsage = (globalThis as any).GPUBufferUsage
  const buffer = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buffer, 0, data)
  return buffer
}

export async function runNTupleGpuBenchmark(preset: NTuplePresetId, requestedBoards = 32768): Promise<NTupleGpuBenchmarkResult> {
  const nav = globalThis.navigator as Navigator & { gpu?: any }
  if (!nav.gpu) {
    return {
      status: 'unavailable',
      preset,
      boards: 0,
      cpuMs: 0,
      gpuMs: 0,
      gpuComputeMs: 0,
      gpuReadbackMs: 0,
      cpuBoardsPerSec: 0,
      gpuBoardsPerSec: 0,
      speedup: 0,
      detail: 'WebGPU is not exposed by this browser. Try Chrome/Edge with hardware acceleration enabled.',
      checksumDiff: 0,
    }
  }

  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No high-performance WebGPU adapter available')
    const device = await adapter.requestDevice()
    const presetDef = NTUPLE_PRESETS[preset]
    const net = new NTupleNetwork(presetDef.patterns, presetDef.symmetric, 0)
    fillSyntheticWeights(net)
    const boards = Math.max(1024, Math.min(requestedBoards, preset === 'expert' ? 4096 : 65536))
    const boardBytes = makeBoards(boards)
    const boardWords = new Uint32Array(boardBytes.length)
    for (let i = 0; i < boardBytes.length; i++) boardWords[i] = boardBytes[i]

    const cpu = cpuBenchmark(net, boardBytes, boards)
    const { weights, offsets } = flattenTables(net)
    const GPUBufferUsage = (globalThis as any).GPUBufferUsage
    const GPUMapMode = (globalThis as any).GPUMapMode
    const boardsBuf = createBuffer(device, boardWords, GPUBufferUsage.STORAGE)
    const weightsBuf = createBuffer(device, weights, GPUBufferUsage.STORAGE)
    const offsetsBuf = createBuffer(device, offsets, GPUBufferUsage.STORAGE)
    const lensBuf = createBuffer(device, Uint32Array.from(net.tupleLens), GPUBufferUsage.STORAGE)
    const patternBuf = createBuffer(device, new Uint32Array(net.viewPattern), GPUBufferUsage.STORAGE)
    const viewOffsetsBuf = createBuffer(device, new Uint32Array(net.viewOffsets), GPUBufferUsage.STORAGE)
    const cellsBuf = createBuffer(device, new Uint32Array(net.viewCells), GPUBufferUsage.STORAGE)
    const outBuf = device.createBuffer({ size: boards * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const dimsBuf = createBuffer(device, new Uint32Array([boards, net.totalViews, 0, 0]), GPUBufferUsage.UNIFORM)
    const readBuf = device.createBuffer({ size: boards * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const module = device.createShaderModule({ code: wgsl })
    const pipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'valueKernel' } })

    const t0 = performance.now()
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipe)
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boardsBuf } },
        { binding: 1, resource: { buffer: weightsBuf } },
        { binding: 2, resource: { buffer: offsetsBuf } },
        { binding: 3, resource: { buffer: lensBuf } },
        { binding: 4, resource: { buffer: patternBuf } },
        { binding: 5, resource: { buffer: viewOffsetsBuf } },
        { binding: 6, resource: { buffer: cellsBuf } },
        { binding: 7, resource: { buffer: outBuf } },
        { binding: 8, resource: { buffer: dimsBuf } },
      ],
    }))
    pass.dispatchWorkgroups(Math.ceil(boards / 64))
    pass.end()
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, boards * 4)
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    const computeMs = performance.now() - t0
    const readT0 = performance.now()
    await readBuf.mapAsync(GPUMapMode.READ)
    const values = new Float32Array(readBuf.getMappedRange().slice(0))
    readBuf.unmap()
    const readbackMs = performance.now() - readT0
    const gpuMs = performance.now() - t0
    let gpuChecksum = 0
    for (let i = 0; i < values.length; i++) gpuChecksum += values[i]
    const info = (adapter as { info?: { vendor?: string; architecture?: string; device?: string } }).info
    const detail = [info?.vendor, info?.architecture, info?.device].filter(Boolean).join(' ') || 'WebGPU high-performance adapter'

    return {
      status: 'ok',
      preset,
      boards,
      cpuMs: cpu.ms,
      gpuMs,
      gpuComputeMs: computeMs,
      gpuReadbackMs: readbackMs,
      cpuBoardsPerSec: (boards / cpu.ms) * 1000,
      gpuBoardsPerSec: (boards / gpuMs) * 1000,
      speedup: cpu.ms / gpuMs,
      detail,
      checksumDiff: Math.abs(cpu.checksum - gpuChecksum),
    }
  } catch (err) {
    return {
      status: 'error',
      preset,
      boards: 0,
      cpuMs: 0,
      gpuMs: 0,
      gpuComputeMs: 0,
      gpuReadbackMs: 0,
      cpuBoardsPerSec: 0,
      gpuBoardsPerSec: 0,
      speedup: 0,
      detail: err instanceof Error ? err.message : String(err),
      checksumDiff: 0,
    }
  }
}


