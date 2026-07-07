// The 8 board symmetries (dihedral group of the square). Each map answers:
// cell i of the transformed board reads from cell map[i] of the original.
import type { Board } from '../../engine/board'

function identityMap(): number[] {
  return Array.from({ length: 16 }, (_, i) => i)
}

/** Compose a 90° clockwise rotation onto an existing map. */
function rotateMap(m: readonly number[]): number[] {
  const out = new Array<number>(16)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) out[r * 4 + c] = m[(3 - c) * 4 + r]
  }
  return out
}

/** Compose a horizontal mirror onto an existing map. */
function flipMap(m: readonly number[]): number[] {
  const out = new Array<number>(16)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) out[r * 4 + c] = m[r * 4 + (3 - c)]
  }
  return out
}

export const SYMMETRY_MAPS: readonly (readonly number[])[] = (() => {
  const maps: number[][] = []
  let m = identityMap()
  for (let k = 0; k < 4; k++) {
    maps.push(m)
    m = rotateMap(m)
  }
  let f = flipMap(identityMap())
  for (let k = 0; k < 4; k++) {
    maps.push(f)
    f = rotateMap(f)
  }
  return maps
})()

export function applySymmetry(board: Board, map: readonly number[], out: Board): void {
  for (let i = 0; i < 16; i++) out[i] = board[map[i]]
}

/**
 * Expand a pattern into its symmetry views: the cell lists to read from the
 * ORIGINAL board so that each read equals evaluating the pattern on one
 * transformed board. Exact duplicate ordered lists are dropped.
 */
export function expandPattern(pattern: readonly number[], symmetric: boolean): number[][] {
  if (!symmetric) return [Array.from(pattern)]
  const views: number[][] = []
  const seen = new Set<string>()
  for (const map of SYMMETRY_MAPS) {
    const cells = pattern.map((c) => map[c])
    const key = cells.join(',')
    if (!seen.has(key)) {
      seen.add(key)
      views.push(cells)
    }
  }
  return views
}
