// Hand-rolled Float32Array MLP with ReLU hidden layers, linear output, and
// Adam. ~100k params — a framework would add megabytes for zero benefit.
import type { Rand } from '../../engine/rng'

const BETA1 = 0.9
const BETA2 = 0.999
const EPS = 1e-8

export class MLP {
  readonly sizes: number[]
  readonly W: Float32Array[] = []
  readonly b: Float32Array[] = []
  private readonly gW: Float32Array[] = []
  private readonly gB: Float32Array[] = []
  private readonly mW: Float32Array[] = []
  private readonly vW: Float32Array[] = []
  private readonly mB: Float32Array[] = []
  private readonly vB: Float32Array[] = []
  private readonly scratch: Float32Array[] = []
  adamT = 0

  constructor(sizes: number[], rand?: Rand) {
    this.sizes = sizes
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l]
      const nOut = sizes[l + 1]
      const W = new Float32Array(nOut * nIn)
      if (rand) {
        // He-normal init via Box-Muller.
        const std = Math.sqrt(2 / nIn)
        for (let i = 0; i < W.length; i += 2) {
          const u1 = Math.max(rand(), 1e-12)
          const u2 = rand()
          const m = Math.sqrt(-2 * Math.log(u1))
          W[i] = m * Math.cos(2 * Math.PI * u2) * std
          if (i + 1 < W.length) W[i + 1] = m * Math.sin(2 * Math.PI * u2) * std
        }
      }
      this.W.push(W)
      this.b.push(new Float32Array(nOut))
      this.gW.push(new Float32Array(nOut * nIn))
      this.gB.push(new Float32Array(nOut))
      this.mW.push(new Float32Array(nOut * nIn))
      this.vW.push(new Float32Array(nOut * nIn))
      this.mB.push(new Float32Array(nOut))
      this.vB.push(new Float32Array(nOut))
      this.scratch.push(new Float32Array(nIn))
    }
  }

  /** Allocate activation storage for forward passes: acts[0] holds the input. */
  makeActs(): Float32Array[] {
    return this.sizes.map((n) => new Float32Array(n))
  }

  /** acts[0] must already contain the input. Returns acts[last] (linear output). */
  forward(acts: Float32Array[]): Float32Array {
    const L = this.W.length
    for (let l = 0; l < L; l++) {
      const aIn = acts[l]
      const aOut = acts[l + 1]
      const nIn = this.sizes[l]
      const nOut = this.sizes[l + 1]
      const W = this.W[l]
      const b = this.b[l]
      const relu = l < L - 1
      for (let o = 0; o < nOut; o++) {
        let s = b[o]
        const base = o * nIn
        for (let i = 0; i < nIn; i++) s += W[base + i] * aIn[i]
        aOut[o] = relu && s < 0 ? 0 : s
      }
    }
    return acts[L]
  }

  /** Accumulate gradients for one sample. `dOut` = dLoss/dOutput. */
  backward(acts: Float32Array[], dOut: Float32Array): void {
    let dz: Float32Array = dOut
    for (let l = this.W.length - 1; l >= 0; l--) {
      const aIn = acts[l]
      const nIn = this.sizes[l]
      const nOut = this.sizes[l + 1]
      const W = this.W[l]
      const gW = this.gW[l]
      const gB = this.gB[l]
      for (let o = 0; o < nOut; o++) {
        const d = dz[o]
        if (d === 0) continue
        gB[o] += d
        const base = o * nIn
        for (let i = 0; i < nIn; i++) gW[base + i] += d * aIn[i]
      }
      if (l > 0) {
        const dPrev = this.scratch[l]
        for (let i = 0; i < nIn; i++) {
          // ReLU derivative from the post-activation value.
          if (aIn[i] <= 0) {
            dPrev[i] = 0
            continue
          }
          let s = 0
          for (let o = 0; o < nOut; o++) s += W[o * nIn + i] * dz[o]
          dPrev[i] = s
        }
        dz = dPrev
      }
    }
  }

  zeroGrads(): void {
    for (let l = 0; l < this.W.length; l++) {
      this.gW[l].fill(0)
      this.gB[l].fill(0)
    }
  }

  /** Adam step over accumulated grads; `scale` is typically 1/batchSize. */
  adamStep(lr: number, scale: number): void {
    this.adamT++
    const c1 = 1 - Math.pow(BETA1, this.adamT)
    const c2 = 1 - Math.pow(BETA2, this.adamT)
    for (let l = 0; l < this.W.length; l++) {
      this.adamArray(this.W[l], this.gW[l], this.mW[l], this.vW[l], lr, scale, c1, c2)
      this.adamArray(this.b[l], this.gB[l], this.mB[l], this.vB[l], lr, scale, c1, c2)
    }
  }

  private adamArray(
    w: Float32Array,
    g: Float32Array,
    m: Float32Array,
    v: Float32Array,
    lr: number,
    scale: number,
    c1: number,
    c2: number,
  ): void {
    for (let i = 0; i < w.length; i++) {
      const grad = g[i] * scale
      const mi = BETA1 * m[i] + (1 - BETA1) * grad
      const vi = BETA2 * v[i] + (1 - BETA2) * grad * grad
      m[i] = mi
      v[i] = vi
      w[i] -= (lr * (mi / c1)) / (Math.sqrt(vi / c2) + EPS)
    }
  }

  copyWeightsFrom(other: MLP): void {
    for (let l = 0; l < this.W.length; l++) {
      this.W[l].set(other.W[l])
      this.b[l].set(other.b[l])
    }
  }

  get paramCount(): number {
    let n = 0
    for (let l = 0; l < this.W.length; l++) n += this.W[l].length + this.b[l].length
    return n
  }

  /** Weights + Adam state, in a fixed order the restore path mirrors. */
  serializeBuffers(): ArrayBuffer[] {
    const out: ArrayBuffer[] = []
    const push = (arrs: Float32Array[]) => {
      for (const a of arrs) out.push(a.buffer.slice(0) as ArrayBuffer)
    }
    push(this.W)
    push(this.b)
    push(this.mW)
    push(this.vW)
    push(this.mB)
    push(this.vB)
    return out
  }

  restoreBuffers(buffers: ArrayBuffer[]): void {
    const L = this.W.length
    if (buffers.length !== L * 6) {
      throw new Error(`MLP snapshot has ${buffers.length} buffers, expected ${L * 6}`)
    }
    const groups = [this.W, this.b, this.mW, this.vW, this.mB, this.vB]
    let k = 0
    for (const group of groups) {
      for (const arr of group) arr.set(new Float32Array(buffers[k++]))
    }
  }
}
