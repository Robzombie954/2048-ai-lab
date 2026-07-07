// Board -> network input: 16 cells x 16 one-hot exponent channels.
// (A scalar log2 encoding starves the net — one-hot is the standard fix.)
import type { Board } from '../../engine/board'

export const DQN_INPUT = 256

export function encodeBoard(board: Board, out: Float32Array): void {
  out.fill(0)
  encodeBoardAt(board, out, 0)
}

export function encodeBoardAt(board: Board, out: Float32Array, offset: number): void {
  for (let i = 0; i < DQN_INPUT; i++) out[offset + i] = 0
  for (let i = 0; i < 16; i++) out[offset + i * 16 + board[i]] = 1
}
