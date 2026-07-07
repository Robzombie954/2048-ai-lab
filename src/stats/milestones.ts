// First-time tile milestones: 128 (exp 7) through 32768 (exp 15).
import type { Milestone } from '../shared/types'

export const MILESTONE_MIN_EXP = 7

export class MilestoneTracker {
  private seen: Set<number>

  constructor(existing: Milestone[] = []) {
    this.seen = new Set(existing.map((m) => m.exp))
  }

  /** Returns a new milestone when `exp` (≥128) appears for the first time. */
  check(exp: number, game: number, totalMoves: number, wallMs: number): Milestone | null {
    if (exp < MILESTONE_MIN_EXP || this.seen.has(exp)) return null
    this.seen.add(exp)
    return { exp, game, totalMoves, wallMs, at: Date.now() }
  }

  reset(existing: Milestone[]): void {
    this.seen = new Set(existing.map((m) => m.exp))
  }
}
