import type { ModelConfig, NTupleConfig, DQNConfig, NTuplePresetId } from '../shared/types'
import type { Agent } from './types'
import { NTupleAgent } from './ntuple/tdLearner'
import { NTUPLE_PRESETS } from './ntuple/patterns'
import { DQNAgent } from './dqn/dqnAgent'

export function createAgent(config: ModelConfig): Agent {
  return config.kind === 'ntuple' ? new NTupleAgent(config) : new DQNAgent(config)
}

export function restoreAgent(metaJson: string, buffers: ArrayBuffer[]): Agent {
  const meta = JSON.parse(metaJson) as { config?: ModelConfig }
  if (meta.config?.kind === 'ntuple') return NTupleAgent.restore(metaJson, buffers)
  if (meta.config?.kind === 'dqn') return DQNAgent.restore(metaJson, buffers)
  throw new Error('unrecognized agent snapshot')
}

export function defaultNTupleConfig(preset: NTuplePresetId = 'starter'): NTupleConfig {
  const p = NTUPLE_PRESETS[preset]
  return {
    kind: 'ntuple',
    preset,
    alpha: p.defaultAlpha,
    tc: p.defaultTc,
    optimisticInit: 0,
    planningDepth: 1,
    gradeStrength: 1,
  }
}

const PRESET_LADDER: NTuplePresetId[] = ['starter', 'balanced', 'expert']

/** The next larger n-tuple preset, or null if already at the top (Expert). */
export function nextPresetUp(preset: NTuplePresetId): NTuplePresetId | null {
  const i = PRESET_LADDER.indexOf(preset)
  return i >= 0 && i < PRESET_LADDER.length - 1 ? PRESET_LADDER[i + 1] : null
}

export function defaultDQNConfig(): DQNConfig {
  return {
    kind: 'dqn',
    hidden: [128, 64],
    lr: 5e-4,
    gamma: 0.99,
    epsStart: 1.0,
    epsEnd: 0.02,
    epsDecayMoves: 50_000,
    replaySize: 20_000,
    batchSize: 32,
    trainFreq: 4,
    targetSync: 1_000,
    shaping: { mode: 'scaled', survivalBonus: false, terminalPenalty: true },
    gradeStrength: 1,
  }
}

/** Estimated in-memory footprint for the wizard's live readout. */
export function estimateModelBytes(config: ModelConfig): number {
  if (config.kind === 'ntuple') {
    const preset = NTUPLE_PRESETS[config.preset]
    let weights = 0
    for (const p of preset.patterns) weights += 1 << (4 * p.length)
    return weights * 4 * (config.tc ? 3 : 1)
  }
  const sizes = [256, ...config.hidden, 4]
  let params = 0
  for (let l = 0; l < sizes.length - 1; l++) params += sizes[l] * sizes[l + 1] + sizes[l + 1]
  return params * 4 * 3 + config.replaySize * 40
}

