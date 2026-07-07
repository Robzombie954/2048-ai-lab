// Board = 16 exponents, row-major. board[r*4+c], 0 = empty.
export type Board = Uint8Array

export function newBoard(): Board {
  return new Uint8Array(16)
}

export function cloneBoard(b: Board): Board {
  return new Uint8Array(b)
}

export function copyBoard(src: Board, dst: Board): void {
  dst.set(src)
}

export function boardsEqual(a: Board, b: Board): boolean {
  for (let i = 0; i < 16; i++) if (a[i] !== b[i]) return false
  return true
}

export function countEmpty(b: Board): number {
  let n = 0
  for (let i = 0; i < 16; i++) if (b[i] === 0) n++
  return n
}

export function maxExp(b: Board): number {
  let m = 0
  for (let i = 0; i < 16; i++) if (b[i] > m) m = b[i]
  return m
}

/** Tile face value for an exponent (0 → 0 for empty). */
export function expValue(exp: number): number {
  return exp === 0 ? 0 : 1 << exp
}

export function boardFromValues(values: number[]): Board {
  const b = newBoard()
  for (let i = 0; i < 16; i++) {
    const v = values[i]
    b[i] = v === 0 ? 0 : Math.round(Math.log2(v))
  }
  return b
}

export function boardToValues(b: Board): number[] {
  return Array.from(b, expValue)
}
