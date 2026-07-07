// @ts-nocheck
import { MLP } from './mlp'

const BETA1 = 0.9
const BETA2 = 0.999
const EPS = 1e-8

interface LayerGpu {
  nIn: number
  nOut: number
  W: GPUBuffer
  b: GPUBuffer
  mW: GPUBuffer
  vW: GPUBuffer
  mB: GPUBuffer
  vB: GPUBuffer
  gW: GPUBuffer
  gB: GPUBuffer
}

interface BatchGpuData {
  states: Float32Array
  nexts: Float32Array
  actions: Uint32Array
  rewards: Float32Array
  terminals: Uint32Array
  masks: Uint32Array
  count: number
}

const shader = `
struct ForwardDims { batch:u32, nIn:u32, nOut:u32, relu:u32 }
struct DqDims { batch:u32, gamma:f32, _pad0:u32, _pad1:u32 }
struct DeltaDims { batch:u32, nCur:u32, nNext:u32, _pad0:u32 }
struct GradDims { batch:u32, nIn:u32, nOut:u32, _pad0:u32 }
struct AdamDims { len:u32, t:f32, lr:f32, scale:f32 }

@group(0) @binding(0) var<storage, read> fInput: array<f32>;
@group(0) @binding(1) var<storage, read> fW: array<f32>;
@group(0) @binding(2) var<storage, read> fB: array<f32>;
@group(0) @binding(3) var<storage, read_write> fOut: array<f32>;
@group(0) @binding(4) var<uniform> fDims: ForwardDims;

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = fDims.batch * fDims.nOut;
  if (idx >= total) { return; }
  let s = idx / fDims.nOut;
  let o = idx % fDims.nOut;
  var acc = fB[o];
  let inBase = s * fDims.nIn;
  let wBase = o * fDims.nIn;
  for (var i:u32 = 0u; i < fDims.nIn; i = i + 1u) {
    acc = acc + fW[wBase + i] * fInput[inBase + i];
  }
  if (fDims.relu == 1u && acc < 0.0) { acc = 0.0; }
  fOut[idx] = acc;
}

@group(1) @binding(0) var<storage, read> q: array<f32>;
@group(1) @binding(1) var<storage, read> nextQ: array<f32>;
@group(1) @binding(2) var<storage, read> actions: array<u32>;
@group(1) @binding(3) var<storage, read> rewards: array<f32>;
@group(1) @binding(4) var<storage, read> terminals: array<u32>;
@group(1) @binding(5) var<storage, read> masks: array<u32>;
@group(1) @binding(6) var<storage, read_write> dQ: array<f32>;
@group(1) @binding(7) var<storage, read_write> absErr: array<f32>;
@group(1) @binding(8) var<uniform> dqDims: DqDims;

@compute @workgroup_size(64)
fn dqKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= dqDims.batch) { return; }
  let a = actions[s];
  var best = -3.402823e38;
  let mask = masks[s];
  for (var d:u32 = 0u; d < 4u; d = d + 1u) {
    if ((mask & (1u << d)) != 0u) {
      best = max(best, nextQ[s * 4u + d]);
    }
  }
  var y = rewards[s];
  if (terminals[s] == 0u && best > -3.0e38) { y = y + dqDims.gamma * best; }
  let err = q[s * 4u + a] - y;
  for (var d:u32 = 0u; d < 4u; d = d + 1u) { dQ[s * 4u + d] = 0.0; }
  dQ[s * 4u + a] = clamp(err, -1.0, 1.0);
  absErr[s] = abs(err);
}

@group(2) @binding(0) var<storage, read> curAct: array<f32>;
@group(2) @binding(1) var<storage, read> nextW: array<f32>;
@group(2) @binding(2) var<storage, read> nextDelta: array<f32>;
@group(2) @binding(3) var<storage, read_write> curDelta: array<f32>;
@group(2) @binding(4) var<uniform> deltaDims: DeltaDims;

@compute @workgroup_size(64)
fn hiddenDelta(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = deltaDims.batch * deltaDims.nCur;
  if (idx >= total) { return; }
  if (curAct[idx] <= 0.0) { curDelta[idx] = 0.0; return; }
  let s = idx / deltaDims.nCur;
  let i = idx % deltaDims.nCur;
  var acc = 0.0;
  for (var o:u32 = 0u; o < deltaDims.nNext; o = o + 1u) {
    acc = acc + nextW[o * deltaDims.nCur + i] * nextDelta[s * deltaDims.nNext + o];
  }
  curDelta[idx] = acc;
}

@group(3) @binding(0) var<storage, read> gInput: array<f32>;
@group(3) @binding(1) var<storage, read> gDelta: array<f32>;
@group(3) @binding(2) var<storage, read_write> gWOut: array<f32>;
@group(3) @binding(3) var<storage, read_write> gBOut: array<f32>;
@group(3) @binding(4) var<uniform> gradDims: GradDims;

@compute @workgroup_size(64)
fn gradW(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = gradDims.nIn * gradDims.nOut;
  if (idx >= total) { return; }
  let o = idx / gradDims.nIn;
  let i = idx % gradDims.nIn;
  var acc = 0.0;
  for (var s:u32 = 0u; s < gradDims.batch; s = s + 1u) {
    acc = acc + gDelta[s * gradDims.nOut + o] * gInput[s * gradDims.nIn + i];
  }
  gWOut[idx] = acc;
}

@compute @workgroup_size(64)
fn gradB(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= gradDims.nOut) { return; }
  var acc = 0.0;
  for (var s:u32 = 0u; s < gradDims.batch; s = s + 1u) {
    acc = acc + gDelta[s * gradDims.nOut + o];
  }
  gBOut[o] = acc;
}

@group(4) @binding(0) var<storage, read_write> aW: array<f32>;
@group(4) @binding(1) var<storage, read> aG: array<f32>;
@group(4) @binding(2) var<storage, read_write> aM: array<f32>;
@group(4) @binding(3) var<storage, read_write> aV: array<f32>;
@group(4) @binding(4) var<uniform> adamDims: AdamDims;

@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= adamDims.len) { return; }
  let grad = aG[i] * adamDims.scale;
  let mi = ${BETA1} * aM[i] + ${1 - BETA1} * grad;
  let vi = ${BETA2} * aV[i] + ${1 - BETA2} * grad * grad;
  aM[i] = mi;
  aV[i] = vi;
  let c1 = 1.0 - pow(${BETA1}, adamDims.t);
  let c2 = 1.0 - pow(${BETA2}, adamDims.t);
  aW[i] = aW[i] - (adamDims.lr * (mi / c1)) / (sqrt(vi / c2) + ${EPS});
}
`

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b)
}

function byteLen(elements: number, bytes = 4): number {
  return Math.max(4, elements * bytes)
}

export class WebGpuDqnBackend {
  readonly detail: string
  private readonly device: GPUDevice
  private readonly module: GPUShaderModule
  private readonly forwardPipe: GPUComputePipeline
  private readonly dqPipe: GPUComputePipeline
  private readonly deltaPipe: GPUComputePipeline
  private readonly gradWPipe: GPUComputePipeline
  private readonly gradBPipe: GPUComputePipeline
  private readonly adamPipe: GPUComputePipeline
  private readonly batch: number
  private readonly sizes: number[]
  private readonly layers: LayerGpu[]
  private readonly targetLayers: LayerGpu[]
  private readonly x: GPUBuffer
  private readonly nextX: GPUBuffer
  private readonly a1: GPUBuffer
  private readonly a2: GPUBuffer
  private readonly q: GPUBuffer
  private readonly n1: GPUBuffer
  private readonly n2: GPUBuffer
  private readonly nextQ: GPUBuffer
  private readonly dQ: GPUBuffer
  private readonly dA2: GPUBuffer
  private readonly dA1: GPUBuffer
  private readonly actions: GPUBuffer
  private readonly rewards: GPUBuffer
  private readonly terminals: GPUBuffer
  private readonly masks: GPUBuffer
  private readonly absErr: GPUBuffer
  private readonly dims: GPUBuffer
  private readonly readback: GPUBuffer
  private readonly weightReadback: GPUBuffer
  private readonly weightBytes: number
  private adamT = 0
  private steps = 0

  private constructor(device: GPUDevice, detail: string, sizes: number[], batch: number, online: MLP, target: MLP) {
    this.device = device
    this.detail = detail
    this.sizes = sizes
    this.batch = batch
    this.module = device.createShaderModule({ code: shader })
    this.forwardPipe = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'forward' } })
    this.dqPipe = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'dqKernel' } })
    this.deltaPipe = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'hiddenDelta' } })
    this.gradWPipe = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'gradW' } })
    this.gradBPipe = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'gradB' } })
    this.adamPipe = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'adam' } })

    this.layers = this.makeLayers(online)
    this.targetLayers = this.makeLayers(target)
    this.x = this.storage(byteLen(batch * sizes[0]))
    this.nextX = this.storage(byteLen(batch * sizes[0]))
    this.a1 = this.storage(byteLen(batch * sizes[1]))
    this.a2 = this.storage(byteLen(batch * sizes[2]))
    this.q = this.storage(byteLen(batch * sizes[3]))
    this.n1 = this.storage(byteLen(batch * sizes[1]))
    this.n2 = this.storage(byteLen(batch * sizes[2]))
    this.nextQ = this.storage(byteLen(batch * sizes[3]))
    this.dQ = this.storage(byteLen(batch * sizes[3]))
    this.dA2 = this.storage(byteLen(batch * sizes[2]))
    this.dA1 = this.storage(byteLen(batch * sizes[1]))
    this.actions = this.storage(byteLen(batch))
    this.rewards = this.storage(byteLen(batch))
    this.terminals = this.storage(byteLen(batch))
    this.masks = this.storage(byteLen(batch))
    this.absErr = this.storage(byteLen(batch))
    this.dims = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.readback = device.createBuffer({ size: byteLen(batch), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    this.weightBytes = online.W.reduce((n, w, i) => n + w.byteLength + online.b[i].byteLength, 0)
    this.weightReadback = device.createBuffer({ size: this.weightBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  }

  static async create(sizes: number[], batch: number, online: MLP, target: MLP): Promise<WebGpuDqnBackend | null> {
    if (sizes.length !== 4) return null
    const nav = globalThis.navigator as Navigator & { gpu?: GPU }
    if (!nav.gpu) return null
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) return null
    const device = await adapter.requestDevice()
    const info = (adapter as unknown as { info?: { vendor?: string; architecture?: string; device?: string } }).info
    const bits = [info?.vendor, info?.architecture, info?.device].filter(Boolean).join(' ')
    return new WebGpuDqnBackend(device, bits || 'WebGPU high-performance adapter', sizes, batch, online, target)
  }

  async train(data: BatchGpuData, gamma: number, lr: number, targetSync: boolean): Promise<number> {
    const count = data.count
    if (count <= 0) return 0
    this.device.queue.writeBuffer(this.x, 0, data.states, 0, count * this.sizes[0])
    this.device.queue.writeBuffer(this.nextX, 0, data.nexts, 0, count * this.sizes[0])
    this.device.queue.writeBuffer(this.actions, 0, data.actions, 0, count)
    this.device.queue.writeBuffer(this.rewards, 0, data.rewards, 0, count)
    this.device.queue.writeBuffer(this.terminals, 0, data.terminals, 0, count)
    this.device.queue.writeBuffer(this.masks, 0, data.masks, 0, count)

    const enc = this.device.createCommandEncoder()
    this.forward(enc, this.x, this.layers[0], this.a1, count, true)
    this.forward(enc, this.a1, this.layers[1], this.a2, count, true)
    this.forward(enc, this.a2, this.layers[2], this.q, count, false)
    this.forward(enc, this.nextX, this.targetLayers[0], this.n1, count, true)
    this.forward(enc, this.n1, this.targetLayers[1], this.n2, count, true)
    this.forward(enc, this.n2, this.targetLayers[2], this.nextQ, count, false)
    this.dq(enc, count, gamma)
    this.hiddenDelta(enc, this.a2, this.layers[2].W, this.dQ, this.dA2, count, this.sizes[2], this.sizes[3])
    this.hiddenDelta(enc, this.a1, this.layers[1].W, this.dA2, this.dA1, count, this.sizes[1], this.sizes[2])
    this.grad(enc, this.a2, this.dQ, this.layers[2], count)
    this.grad(enc, this.a1, this.dA2, this.layers[1], count)
    this.grad(enc, this.x, this.dA1, this.layers[0], count)
    this.adamT++
    for (const layer of this.layers) {
      this.adam(enc, layer.W, layer.gW, layer.mW, layer.vW, layer.W.size / 4, lr, 1 / count)
      this.adam(enc, layer.b, layer.gB, layer.mB, layer.vB, layer.b.size / 4, lr, 1 / count)
    }
    enc.copyBufferToBuffer(this.absErr, 0, this.readback, 0, byteLen(count))
    if (targetSync) this.copyOnlineToTarget(enc)
    this.device.queue.submit([enc.finish()])
    await this.device.queue.onSubmittedWorkDone()
    await this.readback.mapAsync(GPUMapMode.READ, 0, byteLen(count))
    const err = new Float32Array(this.readback.getMappedRange(0, byteLen(count)).slice(0))
    this.readback.unmap()
    this.steps++
    let sum = 0
    for (let i = 0; i < count; i++) sum += err[i] || 0
    return sum / count
  }

  async syncWeightsToCpu(online: MLP, target: MLP): Promise<void> {
    const enc = this.device.createCommandEncoder()
    let offset = 0
    for (const layer of this.layers) {
      enc.copyBufferToBuffer(layer.W, 0, this.weightReadback, offset, layer.W.size)
      offset += layer.W.size
      enc.copyBufferToBuffer(layer.b, 0, this.weightReadback, offset, layer.b.size)
      offset += layer.b.size
    }
    this.device.queue.submit([enc.finish()])
    await this.device.queue.onSubmittedWorkDone()
    await this.weightReadback.mapAsync(GPUMapMode.READ, 0, this.weightBytes)
    const bytes = this.weightReadback.getMappedRange(0, this.weightBytes)
    offset = 0
    for (let i = 0; i < online.W.length; i++) {
      online.W[i].set(new Float32Array(bytes.slice(offset, offset + online.W[i].byteLength)))
      offset += online.W[i].byteLength
      online.b[i].set(new Float32Array(bytes.slice(offset, offset + online.b[i].byteLength)))
      offset += online.b[i].byteLength
    }
    this.weightReadback.unmap()
    target.copyWeightsFrom(online)
  }

  get queueDepth(): number {
    return this.steps
  }

  private makeLayers(net: MLP): LayerGpu[] {
    return net.W.map((w, i) => ({
      nIn: net.sizes[i],
      nOut: net.sizes[i + 1],
      W: this.uploadStorage(w),
      b: this.uploadStorage(net.b[i]),
      mW: this.storage(w.byteLength),
      vW: this.storage(w.byteLength),
      mB: this.storage(net.b[i].byteLength),
      vB: this.storage(net.b[i].byteLength),
      gW: this.storage(w.byteLength),
      gB: this.storage(net.b[i].byteLength),
    }))
  }

  private storage(size: number): GPUBuffer {
    return this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  }

  private uploadStorage(data: Float32Array): GPUBuffer {
    const buf = this.storage(data.byteLength)
    this.device.queue.writeBuffer(buf, 0, data)
    return buf
  }

  private writeDims(values: number[]): void {
    this.device.queue.writeBuffer(this.dims, 0, new Float32Array(values))
  }

  private forward(enc: GPUCommandEncoder, input: GPUBuffer, layer: LayerGpu, out: GPUBuffer, batch: number, relu: boolean): void {
    this.device.queue.writeBuffer(this.dims, 0, new Uint32Array([batch, layer.nIn, layer.nOut, relu ? 1 : 0]))
    const pass = enc.beginComputePass()
    pass.setPipeline(this.forwardPipe)
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.forwardPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: layer.W } },
      { binding: 2, resource: { buffer: layer.b } },
      { binding: 3, resource: { buffer: out } },
      { binding: 4, resource: { buffer: this.dims } },
    ] }))
    pass.dispatchWorkgroups(ceilDiv(batch * layer.nOut, 64))
    pass.end()
  }

  private dq(enc: GPUCommandEncoder, batch: number, gamma: number): void {
    this.writeDims([batch, gamma, 0, 0])
    const pass = enc.beginComputePass()
    pass.setPipeline(this.dqPipe)
    pass.setBindGroup(1, this.device.createBindGroup({ layout: this.dqPipe.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: this.q } },
      { binding: 1, resource: { buffer: this.nextQ } },
      { binding: 2, resource: { buffer: this.actions } },
      { binding: 3, resource: { buffer: this.rewards } },
      { binding: 4, resource: { buffer: this.terminals } },
      { binding: 5, resource: { buffer: this.masks } },
      { binding: 6, resource: { buffer: this.dQ } },
      { binding: 7, resource: { buffer: this.absErr } },
      { binding: 8, resource: { buffer: this.dims } },
    ] }))
    pass.dispatchWorkgroups(ceilDiv(batch, 64))
    pass.end()
  }

  private hiddenDelta(enc: GPUCommandEncoder, act: GPUBuffer, nextW: GPUBuffer, nextDelta: GPUBuffer, out: GPUBuffer, batch: number, nCur: number, nNext: number): void {
    this.device.queue.writeBuffer(this.dims, 0, new Uint32Array([batch, nCur, nNext, 0]))
    const pass = enc.beginComputePass()
    pass.setPipeline(this.deltaPipe)
    pass.setBindGroup(2, this.device.createBindGroup({ layout: this.deltaPipe.getBindGroupLayout(2), entries: [
      { binding: 0, resource: { buffer: act } },
      { binding: 1, resource: { buffer: nextW } },
      { binding: 2, resource: { buffer: nextDelta } },
      { binding: 3, resource: { buffer: out } },
      { binding: 4, resource: { buffer: this.dims } },
    ] }))
    pass.dispatchWorkgroups(ceilDiv(batch * nCur, 64))
    pass.end()
  }

  private grad(enc: GPUCommandEncoder, input: GPUBuffer, delta: GPUBuffer, layer: LayerGpu, batch: number): void {
    this.device.queue.writeBuffer(this.dims, 0, new Uint32Array([batch, layer.nIn, layer.nOut, 0]))
    let pass = enc.beginComputePass()
    pass.setPipeline(this.gradWPipe)
    pass.setBindGroup(3, this.device.createBindGroup({ layout: this.gradWPipe.getBindGroupLayout(3), entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: delta } },
      { binding: 2, resource: { buffer: layer.gW } },
      { binding: 3, resource: { buffer: layer.gB } },
      { binding: 4, resource: { buffer: this.dims } },
    ] }))
    pass.dispatchWorkgroups(ceilDiv(layer.nIn * layer.nOut, 64))
    pass.end()
    pass = enc.beginComputePass()
    pass.setPipeline(this.gradBPipe)
    pass.setBindGroup(3, this.device.createBindGroup({ layout: this.gradBPipe.getBindGroupLayout(3), entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: delta } },
      { binding: 2, resource: { buffer: layer.gW } },
      { binding: 3, resource: { buffer: layer.gB } },
      { binding: 4, resource: { buffer: this.dims } },
    ] }))
    pass.dispatchWorkgroups(ceilDiv(layer.nOut, 64))
    pass.end()
  }

  private adam(enc: GPUCommandEncoder, w: GPUBuffer, g: GPUBuffer, m: GPUBuffer, v: GPUBuffer, len: number, lr: number, scale: number): void {
    this.writeDims([len, this.adamT, lr, scale])
    const pass = enc.beginComputePass()
    pass.setPipeline(this.adamPipe)
    pass.setBindGroup(4, this.device.createBindGroup({ layout: this.adamPipe.getBindGroupLayout(4), entries: [
      { binding: 0, resource: { buffer: w } },
      { binding: 1, resource: { buffer: g } },
      { binding: 2, resource: { buffer: m } },
      { binding: 3, resource: { buffer: v } },
      { binding: 4, resource: { buffer: this.dims } },
    ] }))
    pass.dispatchWorkgroups(ceilDiv(len, 64))
    pass.end()
  }

  private copyOnlineToTarget(enc: GPUCommandEncoder): void {
    for (let i = 0; i < this.layers.length; i++) {
      enc.copyBufferToBuffer(this.layers[i].W, 0, this.targetLayers[i].W, 0, this.layers[i].W.size)
      enc.copyBufferToBuffer(this.layers[i].b, 0, this.targetLayers[i].b, 0, this.layers[i].b.size)
    }
  }
}

export type { BatchGpuData }

