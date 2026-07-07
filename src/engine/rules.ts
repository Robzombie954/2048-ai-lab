// The canonical 2048 movement rules. Everything — the LUT-based fast engine,
// the tile-identity trace engine, and the tests — derives from slideLine.

export type Dir = 0 | 1 | 2 | 3
export const UP: Dir = 0
export const RIGHT: Dir = 1
export const DOWN: Dir = 2
export const LEFT: Dir = 3
export const DIRS: readonly Dir[] = [UP, RIGHT, DOWN, LEFT]
export const DIR_NAMES = ['up', 'right', 'down', 'left'] as const
export const DIR_ARROWS = ['↑', '→', '↓', '←'] as const

// Cells are stored as exponents: 0 = empty, 1 = tile 2, ..., 15 = tile 32768.
// 4 bits per cell in the row LUTs caps exponents at 15 — merging two 32768s
// (unreachable in practice) yields another 32768 but still scores 65536.
export const MAX_EXP = 15

export interface LineResult {
  out: [number, number, number, number]
  score: number
  moved: boolean
}

/**
 * Slide a 4-cell line of exponents toward index 0, merging equal adjacent
 * tiles once per move (nearest the edge merges first), exactly like the
 * original game. Score gained = face value of each tile created.
 */
export function slideLine(line: ArrayLike<number>): LineResult {
  const out: [number, number, number, number] = [0, 0, 0, 0]
  let score = 0
  let write = 0
  let mergeable = -1 // slot index eligible to receive a merge
  for (let i = 0; i < 4; i++) {
    const v = line[i]
    if (v === 0) continue
    if (mergeable >= 0 && out[mergeable] === v) {
      out[mergeable] = Math.min(v + 1, MAX_EXP)
      score += 1 << (v + 1)
      mergeable = -1 // one merge per tile per move
    } else {
      out[write] = v
      mergeable = write
      write++
    }
  }
  const moved =
    out[0] !== line[0] || out[1] !== line[1] || out[2] !== line[2] || out[3] !== line[3]
  return { out, score, moved }
}
