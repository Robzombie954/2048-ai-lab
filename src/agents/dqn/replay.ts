// Experience replay ring buffer in packed typed arrays. Manually-graded
// transitions are sampled at ~4× probability so a human grade keeps teaching.
import type { Board } from '../../engine/board'
import type { Rand } from '../../engine/rng'

export class ReplayBuffer {
  readonly cap: number
  size = 0
  private pos = 0
  private readonly states: Uint8Array
  private readonly actions: Uint8Array
  private readonly rewards: Float32Array
  private readonly nexts: Uint8Array
  private readonly terminals: Uint8Array
  private readonly graded: Uint8Array
  private gradedList: number[] = []

  constructor(cap: number) {
    this.cap = cap
    this.states = new Uint8Array(cap * 16)
    this.actions = new Uint8Array(cap)
    this.rewards = new Float32Array(cap)
    this.nexts = new Uint8Array(cap * 16)
    this.terminals = new Uint8Array(cap)
    this.graded = new Uint8Array(cap)
  }

  push(
    state: Board,
    action: number,
    reward: number,
    next: Board,
    terminal: boolean,
    isGraded: boolean,
  ): number {
    const i = this.pos
    this.states.set(state, i * 16)
    this.nexts.set(next, i * 16)
    this.actions[i] = action
    this.rewards[i] = reward
    this.terminals[i] = terminal ? 1 : 0
    this.graded[i] = isGraded ? 1 : 0
    if (isGraded) {
      this.gradedList.push(i)
      if (this.gradedList.length > 1024) this.gradedList.shift()
    }
    this.pos = (this.pos + 1) % this.cap
    if (this.size < this.cap) this.size++
    return i
  }

  stateAt(i: number): Board {
    return this.states.subarray(i * 16, i * 16 + 16)
  }

  nextAt(i: number): Board {
    return this.nexts.subarray(i * 16, i * 16 + 16)
  }

  copyStateInto(i: number, out: Board): void {
    const base = i * 16
    for (let c = 0; c < 16; c++) out[c] = this.states[base + c]
  }

  copyNextInto(i: number, out: Board): void {
    const base = i * 16
    for (let c = 0; c < 16; c++) out[c] = this.nexts[base + c]
  }

  actionAt(i: number): number {
    return this.actions[i]
  }

  rewardAt(i: number): number {
    return this.rewards[i]
  }

  terminalAt(i: number): boolean {
    return this.terminals[i] === 1
  }

  fillSample(rand: Rand, out: Uint32Array, n: number): void {
    for (let i = 0; i < n; i++) out[i] = this.sampleIndex(rand)
  }

  sampleIndex(rand: Rand): number {
    // Prune stale graded entries lazily (overwritten by the ring).
    while (this.gradedList.length > 0) {
      const probe = this.gradedList[0]
      if (this.graded[probe] === 1) break
      this.gradedList.shift()
    }
    const g = this.gradedList.length
    if (g > 0) {
      const pGraded = Math.min(0.5, (4 * g) / (this.size + 3 * g))
      if (rand() < pGraded) {
        const idx = this.gradedList[(rand() * g) | 0]
        if (this.graded[idx] === 1) return idx
      }
    }
    return (rand() * this.size) | 0
  }
}


