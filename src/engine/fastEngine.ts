// Headless LUT-based engine used by the training worker and by agents to
// compute afterstates. Millions of moves per second; zero allocations on
// the hot path (callers provide output boards).
import type { Board } from './board'
import { type Dir, UP, RIGHT, DOWN, LEFT } from './rules'
import {
  LEFT_RESULT,
  LEFT_SCORE,
  LEFT_MOVED,
  RIGHT_RESULT,
  RIGHT_SCORE,
  RIGHT_MOVED,
  packRow,
} from './luts'

export interface MoveResult {
  moved: boolean
  score: number
}

/**
 * Apply `dir` to `board`, writing the post-slide (pre-spawn) afterstate into
 * `out`. `out` may alias `board`. Returns merge score gained and whether
 * anything moved.
 */
export function applyMove(board: Board, dir: Dir, out: Board): MoveResult {
  let score = 0
  let moved = false
  if (dir === LEFT || dir === RIGHT) {
    const result = dir === LEFT ? LEFT_RESULT : RIGHT_RESULT
    const scores = dir === LEFT ? LEFT_SCORE : RIGHT_SCORE
    const movedT = dir === LEFT ? LEFT_MOVED : RIGHT_MOVED
    for (let r = 0; r < 4; r++) {
      const i = r * 4
      const code = packRow(board[i], board[i + 1], board[i + 2], board[i + 3])
      const next = result[code]
      out[i] = next >>> 12
      out[i + 1] = (next >>> 8) & 0xf
      out[i + 2] = (next >>> 4) & 0xf
      out[i + 3] = next & 0xf
      score += scores[code]
      if (movedT[code]) moved = true
    }
  } else {
    // Up = slide toward row 0 = LEFT on column codes; down = RIGHT.
    const result = dir === UP ? LEFT_RESULT : RIGHT_RESULT
    const scores = dir === UP ? LEFT_SCORE : RIGHT_SCORE
    const movedT = dir === UP ? LEFT_MOVED : RIGHT_MOVED
    for (let c = 0; c < 4; c++) {
      const code = packRow(board[c], board[c + 4], board[c + 8], board[c + 12])
      const next = result[code]
      out[c] = next >>> 12
      out[c + 4] = (next >>> 8) & 0xf
      out[c + 8] = (next >>> 4) & 0xf
      out[c + 12] = next & 0xf
      score += scores[code]
      if (movedT[code]) moved = true
    }
  }
  return { moved, score }
}

/** Bitmask of legal directions: bit (1 << dir) set if the move changes the board. */
export function legalMask(board: Board): number {
  let mask = 0
  for (let r = 0; r < 4; r++) {
    const i = r * 4
    const code = packRow(board[i], board[i + 1], board[i + 2], board[i + 3])
    if (LEFT_MOVED[code]) mask |= 1 << LEFT
    if (RIGHT_MOVED[code]) mask |= 1 << RIGHT
  }
  for (let c = 0; c < 4; c++) {
    const code = packRow(board[c], board[c + 4], board[c + 8], board[c + 12])
    if (LEFT_MOVED[code]) mask |= 1 << UP
    if (RIGHT_MOVED[code]) mask |= 1 << DOWN
  }
  return mask
}

export function isGameOver(board: Board): boolean {
  return legalMask(board) === 0
}

export interface SpawnInfo {
  idx: number
  exp: number
}

/**
 * Spawn a tile in a uniformly random empty cell: 2 with p=0.9, 4 with p=0.1
 * (original game odds). Mutates `board`. First rand() picks the cell, second
 * picks the value — the order is part of the determinism contract.
 */
export function spawnTile(board: Board, rand: () => number): SpawnInfo {
  let empties = 0
  for (let i = 0; i < 16; i++) if (board[i] === 0) empties++
  let k = (rand() * empties) | 0
  if (k >= empties) k = empties - 1
  const exp = rand() < 0.9 ? 1 : 2
  for (let i = 0; i < 16; i++) {
    if (board[i] === 0) {
      if (k === 0) {
        board[i] = exp
        return { idx: i, exp }
      }
      k--
    }
  }
  throw new Error('spawnTile called on a full board')
}

/** Reset `board` to a fresh game: cleared with two spawned tiles. */
export function startBoard(board: Board, rand: () => number): [SpawnInfo, SpawnInfo] {
  board.fill(0)
  return [spawnTile(board, rand), spawnTile(board, rand)]
}
