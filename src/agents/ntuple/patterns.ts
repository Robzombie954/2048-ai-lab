// N-tuple pattern presets. Cell indices are row-major board positions.
// Honest sizing: weights are Float32; TC (when enabled) triples memory.
import type { NTuplePresetId } from '../../shared/types'

export interface NTuplePreset {
  id: NTuplePresetId
  name: string
  tagline: string
  patterns: number[][]
  /** Expand each pattern across the 8 board symmetries with shared weights. */
  symmetric: boolean
  defaultAlpha: number
  defaultTc: boolean
}

const rows = [0, 1, 2, 3].map((r) => [r * 4, r * 4 + 1, r * 4 + 2, r * 4 + 3])
const cols = [0, 1, 2, 3].map((c) => [c, c + 4, c + 8, c + 12])
const squares: number[][] = []
for (let r = 0; r < 3; r++) {
  for (let c = 0; c < 3; c++) {
    const i = r * 4 + c
    squares.push([i, i + 1, i + 4, i + 5])
  }
}

export const NTUPLE_PRESETS: Record<NTuplePresetId, NTuplePreset> = {
  // 17 four-tuples: every row, column, and 2×2 square. Closed under symmetry,
  // evaluated directly. 17 × 16⁴ weights = 1.11M (4.5MB). Learns visibly
  // within a minute of turbo.
  starter: {
    id: 'starter',
    name: 'Starter — 17×4-tuple',
    tagline: 'Fast visible learning; plateaus in the 10–30k score range.',
    patterns: [...rows, ...cols, ...squares],
    symmetric: false,
    defaultAlpha: 1.0,
    defaultTc: true,
  },
  // 8 five-tuples (row overhangs + Ls), 8-symmetry weight sharing.
  // 8 × 16⁵ = 8.4M weights (33.5MB).
  balanced: {
    id: 'balanced',
    name: 'Balanced — 8×5-tuple',
    tagline: 'Strong play — regular 2048–4096 tiles after hours of turbo.',
    patterns: [
      [0, 1, 2, 3, 4],
      [4, 5, 6, 7, 8],
      [8, 9, 10, 11, 12],
      [0, 1, 2, 4, 5],
      [4, 5, 6, 8, 9],
      [1, 2, 5, 6, 9],
      [0, 1, 4, 5, 8],
      [5, 6, 9, 10, 13],
    ],
    symmetric: true,
    defaultAlpha: 0.1,
    defaultTc: false,
  },
  // The classic 4×6-tuple configuration (Yeh) that genuinely masters the
  // game. 4 × 16⁶ = 67M weights (268MB) — flagged in the wizard.
  expert: {
    id: 'expert',
    name: 'Expert — 4×6-tuple',
    tagline: 'The literature config: 8192+ tiles possible with planning depth 3.',
    patterns: [
      [0, 1, 2, 3, 4, 5],
      [4, 5, 6, 7, 8, 9],
      [0, 1, 2, 4, 5, 6],
      [4, 5, 6, 8, 9, 10],
    ],
    symmetric: true,
    defaultAlpha: 0.1,
    defaultTc: false,
  },
}
