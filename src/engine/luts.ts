// Row lookup tables: every possible packed row (4 cells × 4 bits = 65,536)
// is precomputed from the canonical slideLine rule. Built once at module
// load in <50ms, ~0.7MB total.
import { slideLine } from './rules'

export const ROW_COUNT = 0x10000

export const LEFT_RESULT = new Uint16Array(ROW_COUNT)
export const LEFT_SCORE = new Uint32Array(ROW_COUNT)
export const LEFT_MOVED = new Uint8Array(ROW_COUNT)
export const RIGHT_RESULT = new Uint16Array(ROW_COUNT)
export const RIGHT_SCORE = new Uint32Array(ROW_COUNT)
export const RIGHT_MOVED = new Uint8Array(ROW_COUNT)

/** Cell 0 lives in the high nibble so a packed row reads left-to-right. */
export function packRow(a: number, b: number, c: number, d: number): number {
  return (a << 12) | (b << 8) | (c << 4) | d
}

{
  const cells = [0, 0, 0, 0]
  const rev = [0, 0, 0, 0]
  for (let row = 0; row < ROW_COUNT; row++) {
    cells[0] = row >>> 12
    cells[1] = (row >>> 8) & 0xf
    cells[2] = (row >>> 4) & 0xf
    cells[3] = row & 0xf

    const left = slideLine(cells)
    LEFT_RESULT[row] = packRow(left.out[0], left.out[1], left.out[2], left.out[3])
    LEFT_SCORE[row] = left.score
    LEFT_MOVED[row] = left.moved ? 1 : 0

    rev[0] = cells[3]
    rev[1] = cells[2]
    rev[2] = cells[1]
    rev[3] = cells[0]
    const right = slideLine(rev)
    RIGHT_RESULT[row] = packRow(right.out[3], right.out[2], right.out[1], right.out[0])
    RIGHT_SCORE[row] = right.score
    RIGHT_MOVED[row] = right.moved ? 1 : 0
  }
}
