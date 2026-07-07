// Tile-identity engine for the animated UI board. Derives movement from the
// same walk semantics as the canonical slideLine rule, but carries persistent
// tile ids and emits slide/merge/spawn events for animation. Proven
// behaviorally identical to the LUT fast engine by the exhaustive test suite.
import { type Board, newBoard } from './board'
import { type Dir, MAX_EXP } from './rules'
import { legalMask, spawnTile, type SpawnInfo } from './fastEngine'
import { splitmix32, randomSeed, type Rand } from './rng'

export interface UITile {
  id: number
  idx: number
  exp: number
}

export interface SlideEvent {
  id: number
  from: number
  to: number
}

export interface MergeEvent {
  survivorId: number
  victimId: number
  at: number
  exp: number // post-merge exponent
}

export interface SpawnEvent {
  id: number
  idx: number
  exp: number
}

export interface MoveEvents {
  scoreGained: number
  slides: SlideEvent[]
  merges: MergeEvent[]
  spawn: SpawnEvent | null
  gameOver: boolean
}

// Walk order per direction: each line lists cell indices starting at the edge
// tiles slide toward.
function buildLineOrders(): number[][][] {
  const rows = [0, 1, 2, 3].map((r) => [r * 4, r * 4 + 1, r * 4 + 2, r * 4 + 3])
  const cols = [0, 1, 2, 3].map((c) => [c, c + 4, c + 8, c + 12])
  const rev = (lines: number[][]) => lines.map((l) => [...l].reverse())
  // Dir order: 0=up, 1=right, 2=down, 3=left
  return [cols, rev(rows), rev(cols), rows]
}
const LINE_ORDERS = buildLineOrders()

export interface TracedLine {
  placed: { id: number; exp: number; slot: number }[]
  merges: { survivorId: number; victimId: number; slot: number; exp: number }[]
  moves: { id: number; fromPos: number; toSlot: number }[]
  score: number
}

/**
 * The id-carrying twin of slideLine: entries are the line's non-empty tiles in
 * walk order (pos = position along the line). Same one-merge-per-tile,
 * nearest-edge-first semantics.
 */
export function slideLineWithIds(
  entries: { id: number; exp: number; pos: number }[],
): TracedLine {
  const placed: { id: number; exp: number; slot: number }[] = []
  const merges: TracedLine['merges'] = []
  const moves: TracedLine['moves'] = []
  let score = 0
  let mergeable = -1 // index into `placed` eligible to receive a merge
  for (const e of entries) {
    if (mergeable >= 0 && placed[mergeable].exp === e.exp) {
      const survivor = placed[mergeable]
      const newExp = Math.min(e.exp + 1, MAX_EXP)
      score += 1 << (e.exp + 1)
      survivor.exp = newExp
      merges.push({ survivorId: survivor.id, victimId: e.id, slot: survivor.slot, exp: newExp })
      moves.push({ id: e.id, fromPos: e.pos, toSlot: survivor.slot })
      mergeable = -1
    } else {
      const slot = placed.length
      placed.push({ id: e.id, exp: e.exp, slot })
      moves.push({ id: e.id, fromPos: e.pos, toSlot: slot })
      mergeable = slot
    }
  }
  return { placed, merges, moves, score }
}

export class TraceGame {
  readonly board: Board = newBoard()
  score = 0
  over = false
  seed = 0
  moveCount = 0
  private cells: (UITile | null)[] = new Array(16).fill(null)
  private nextId = 1
  private rand: Rand = splitmix32(0)

  constructor(seed?: number) {
    this.reset(seed)
  }

  reset(seed: number = randomSeed()): SpawnEvent[] {
    this.seed = seed
    this.rand = splitmix32(seed)
    this.board.fill(0)
    this.cells.fill(null)
    this.score = 0
    this.over = false
    this.moveCount = 0
    this.nextId = 1
    const spawns: SpawnEvent[] = []
    for (let i = 0; i < 2; i++) {
      const s = spawnTile(this.board, this.rand)
      spawns.push(this.addTile(s))
    }
    return spawns
  }

  /** Current tiles (for rendering). */
  tiles(): UITile[] {
    const out: UITile[] = []
    for (const t of this.cells) if (t) out.push(t)
    return out
  }

  legalMask(): number {
    return legalMask(this.board)
  }

  /**
   * Apply a move. With `externalSpawn` provided (including null for
   * "no spawn"), mirrors a game driven elsewhere (the training worker);
   * without it, spawns from this game's own seeded RNG (human play).
   * Returns null if the move is illegal.
   */
  move(dir: Dir, externalSpawn?: SpawnInfo | null): MoveEvents | null {
    if (this.over) return null
    const lines = LINE_ORDERS[dir]
    const slides: SlideEvent[] = []
    const merges: MergeEvent[] = []
    const newCells: (UITile | null)[] = new Array(16).fill(null)
    let gained = 0

    for (const line of lines) {
      const entries: { id: number; exp: number; pos: number }[] = []
      for (let p = 0; p < 4; p++) {
        const t = this.cells[line[p]]
        if (t) entries.push({ id: t.id, exp: t.exp, pos: p })
      }
      const traced = slideLineWithIds(entries)
      gained += traced.score
      for (const m of traced.moves) {
        const from = line[m.fromPos]
        const to = line[m.toSlot]
        if (from !== to) slides.push({ id: m.id, from, to })
      }
      for (const mg of traced.merges) {
        merges.push({
          survivorId: mg.survivorId,
          victimId: mg.victimId,
          at: line[mg.slot],
          exp: mg.exp,
        })
      }
      for (const pl of traced.placed) {
        const idx = line[pl.slot]
        newCells[idx] = { id: pl.id, idx, exp: pl.exp }
      }
    }

    if (slides.length === 0 && merges.length === 0) return null

    for (let i = 0; i < 16; i++) {
      this.cells[i] = newCells[i]
      this.board[i] = newCells[i] ? newCells[i]!.exp : 0
    }
    this.score += gained
    this.moveCount++

    let spawnEv: SpawnEvent | null = null
    if (externalSpawn === undefined) {
      spawnEv = this.addTile(spawnTile(this.board, this.rand))
    } else if (externalSpawn !== null) {
      this.board[externalSpawn.idx] = externalSpawn.exp
      spawnEv = this.addTile(externalSpawn)
    }

    this.over = legalMask(this.board) === 0
    return { scoreGained: gained, slides, merges, spawn: spawnEv, gameOver: this.over }
  }

  /** Hard resync to an externally-supplied state (fresh ids, teleport render). */
  forceState(board: Board, score: number): void {
    this.board.set(board)
    this.score = score
    for (let i = 0; i < 16; i++) {
      this.cells[i] = board[i] ? { id: this.nextId++, idx: i, exp: board[i] } : null
    }
    this.over = legalMask(this.board) === 0
  }

  private addTile(s: SpawnInfo): SpawnEvent {
    const tile: UITile = { id: this.nextId++, idx: s.idx, exp: s.exp }
    this.cells[s.idx] = tile
    return { id: tile.id, idx: s.idx, exp: s.exp }
  }
}
