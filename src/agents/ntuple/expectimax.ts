// Optional planning at inference: exact expectimax over the spawn chance
// node (every empty cell × {2: 0.9, 4: 0.1}), with leaves valued ONLY by the
// learned value network. plies = additional (spawn, move) levels beyond the
// direct V(afterstate) read; plies = 0 is the pure learned policy.
import type { Board } from '../../engine/board'
import { DIRS } from '../../engine/rules'
import { applyMove, legalMask } from '../../engine/fastEngine'
import type { NTupleNetwork } from './network'

const SPAWNS: readonly (readonly [number, number])[] = [
  [1, 0.9],
  [2, 0.1],
]

// Scratch boards per recursion level (max plies is small).
const POOL: Board[] = Array.from({ length: 12 }, () => new Uint8Array(16))

export function expectimaxValue(
  net: NTupleNetwork,
  after: Board,
  plies: number,
  level = 0,
): number {
  if (plies <= 0) return net.value(after)
  const child = POOL[level * 2]
  const childAfter = POOL[level * 2 + 1]
  let empties = 0
  for (let i = 0; i < 16; i++) if (after[i] === 0) empties++
  if (empties === 0) return net.value(after) // unreachable for a true afterstate
  let expected = 0
  for (let i = 0; i < 16; i++) {
    if (after[i] !== 0) continue
    for (const [exp, p] of SPAWNS) {
      child.set(after)
      child[i] = exp
      const mask = legalMask(child)
      let best = 0 // dead position is worth exactly its 0 future reward
      if (mask !== 0) {
        best = -Infinity
        for (const dir of DIRS) {
          if (!(mask & (1 << dir))) continue
          const r = applyMove(child, dir, childAfter)
          const v = r.score + expectimaxValue(net, childAfter, plies - 1, level + 1)
          if (v > best) best = v
        }
      }
      expected += (p / empties) * best
    }
  }
  return expected
}
