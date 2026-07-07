import type { AnnealConfig } from '../../shared/types'

export const DEFAULT_ANNEAL_HALFLIFE = 4000
export const DEFAULT_ANNEAL_FLOOR = 0.1

/**
 * Effective learning-rate multiplier at `gamesSeen`. Decays by half every
 * `halfLifeGames` past `startGame`, clamped to `floor`. Returns 1 when no
 * schedule is set, so annealing-off behaviour is byte-identical.
 */
export function annealMultiplier(gamesSeen: number, anneal: AnnealConfig | null | undefined): number {
  if (!anneal || !Number.isFinite(anneal.halfLifeGames) || anneal.halfLifeGames <= 0) return 1
  const t = Math.max(0, gamesSeen - anneal.startGame)
  const mult = Math.pow(0.5, t / anneal.halfLifeGames)
  return Math.max(anneal.floor, Math.min(1, mult))
}
