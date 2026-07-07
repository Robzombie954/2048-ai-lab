// N-tuple value network: V(board) = Σ over pattern views of one Float32
// table read, indexed by the view's cell exponents packed base-16.
// Evaluation is pure gather+sum — no heuristics, only learned weights.
import type { Board } from '../../engine/board'
import { expandPattern } from './symmetry'

export class NTupleNetwork {
  readonly tables: Float32Array[]
  readonly tupleLens: number[]
  /** viewPattern[v] = pattern (table) index of view v. */
  readonly viewPattern: Int32Array
  /** Flattened cell lists; view v occupies viewOffsets[v] .. +tupleLens[viewPattern[v]]. */
  readonly viewCells: Uint8Array
  readonly viewOffsets: Int32Array
  readonly totalViews: number

  constructor(patterns: readonly (readonly number[])[], symmetric: boolean, initTotal = 0) {
    this.tables = []
    this.tupleLens = []
    const viewPattern: number[] = []
    const viewOffsets: number[] = []
    const cells: number[] = []
    for (let p = 0; p < patterns.length; p++) {
      const len = patterns[p].length
      this.tupleLens.push(len)
      this.tables.push(new Float32Array(1 << (4 * len)))
      for (const view of expandPattern(patterns[p], symmetric)) {
        viewPattern.push(p)
        viewOffsets.push(cells.length)
        cells.push(...view)
      }
    }
    this.viewPattern = Int32Array.from(viewPattern)
    this.viewOffsets = Int32Array.from(viewOffsets)
    this.viewCells = Uint8Array.from(cells)
    this.totalViews = viewPattern.length
    if (initTotal !== 0) {
      const perWeight = initTotal / this.totalViews
      for (const t of this.tables) t.fill(perWeight)
    }
  }

  value(board: Board): number {
    let v = 0
    for (let vi = 0; vi < this.totalViews; vi++) {
      const p = this.viewPattern[vi]
      const len = this.tupleLens[p]
      const off = this.viewOffsets[vi]
      let idx = 0
      for (let j = 0; j < len; j++) idx = (idx << 4) | board[this.viewCells[off + j]]
      v += this.tables[p][idx]
    }
    return v
  }

  /** Fill `out` (length ≥ totalViews) with each view's table index for `board`. */
  gatherIndices(board: Board, out: Int32Array): void {
    for (let vi = 0; vi < this.totalViews; vi++) {
      const p = this.viewPattern[vi]
      const len = this.tupleLens[p]
      const off = this.viewOffsets[vi]
      let idx = 0
      for (let j = 0; j < len; j++) idx = (idx << 4) | board[this.viewCells[off + j]]
      out[vi] = idx
    }
  }

  get paramCount(): number {
    let n = 0
    for (const t of this.tables) n += t.length
    return n
  }
}
