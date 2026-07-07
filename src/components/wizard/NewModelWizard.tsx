import { useEffect, useMemo, useState } from 'react'
import { setLab, useLabStore } from '../../state/labStore'
import { bridge } from '../../state/workerBridge'
import type { DQNConfig, NTupleConfig, NTuplePresetId } from '../../shared/types'
import { NTUPLE_PRESETS } from '../../agents/ntuple/patterns'
import { defaultDQNConfig, defaultNTupleConfig, estimateModelBytes } from '../../agents/factory'
import { formatBytes, formatCompact } from '../../lib/format'

type Kind = 'ntuple' | 'dqn'
type Step = 'kind' | 'arch' | 'tune' | 'review'

const HIDDEN_PRESETS: { label: string; layers: number[] }[] = [
  { label: 'Compact — 128×64', layers: [128, 64] },
  { label: 'Standard — 256×128', layers: [256, 128] },
  { label: 'Wide — 512×256', layers: [512, 256] },
]

export function NewModelWizard() {
  const open = useLabStore((s) => s.wizardOpen)
  const seed = useLabStore((s) => s.wizardSeed)
  const [step, setStep] = useState<Step>('kind')
  const [kind, setKind] = useState<Kind>('ntuple')
  const [nt, setNt] = useState<NTupleConfig>(defaultNTupleConfig('starter'))
  const [dqn, setDqn] = useState<DQNConfig>(defaultDQNConfig())
  const [name, setName] = useState('')

  const config = kind === 'ntuple' ? nt : dqn
  const memory = useMemo(() => estimateModelBytes(config), [config])

  // "Level up" pre-seeds the wizard with the next preset up + optimistic init
  // (speeds the early flailing phase) and jumps straight to review.
  useEffect(() => {
    if (open && seed) {
      setKind('ntuple')
      setNt({ ...defaultNTupleConfig(seed.preset), optimisticInit: 80_000 })
      setName(`${seed.fromName} Mk II`)
      setStep('review')
    }
  }, [open, seed])

  if (!open) return null

  const close = () => {
    setLab({ wizardOpen: false, wizardSeed: null })
    setStep('kind')
    setName('')
  }

  const create = () => {
    const finalName = name.trim() || `${kind === 'ntuple' ? 'N-Tuple' : 'DQN'} ${new Date().toLocaleDateString()}`
    if (seed?.autostart) setLab({ autostartOnLoad: true })
    bridge.createModel(finalName, config)
    close()
  }

  const steps: Step[] = ['kind', 'arch', 'tune', 'review']
  const stepIdx = steps.indexOf(step)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm" onClick={close}>
      <div
        className="flash-in max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-2xl border border-white/10 bg-surface-1 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-zinc-100">New model</h2>
          <div className="flex items-center gap-1.5">
            {steps.map((s, i) => (
              <div
                key={s}
                className={`h-1.5 rounded-full transition-all ${
                  i === stepIdx ? 'w-6 bg-gold-500' : i < stepIdx ? 'w-3 bg-gold-500/40' : 'w-3 bg-white/10'
                }`}
              />
            ))}
          </div>
        </div>

        {step === 'kind' && (
          <div className="space-y-3">
            <p className="text-sm leading-relaxed text-zinc-400">
              Both start knowing <span className="text-zinc-200">nothing</span> — no strategy is
              programmed in. They learn purely from playing.
            </p>
            <KindCard
              selected={kind === 'ntuple'}
              onClick={() => setKind('ntuple')}
              title="N-Tuple Network"
              badge="the fast learner"
              body="Millions of tiny pattern weights trained by temporal-difference learning. Visibly improves within minutes and can genuinely master the game."
            />
            <KindCard
              selected={kind === 'dqn'}
              onClick={() => setKind('dqn')}
              title="Neural Network (DQN)"
              badge="the slow real thing"
              body="A from-scratch deep Q-network — the classic. Learns slowly over hours or days, and watching it crawl upward is the honest experience."
            />
          </div>
        )}

        {step === 'arch' && kind === 'ntuple' && (
          <div className="space-y-3">
            {(Object.keys(NTUPLE_PRESETS) as NTuplePresetId[]).map((id) => {
              const p = NTUPLE_PRESETS[id]
              const bytes = estimateModelBytes({ ...nt, preset: id, tc: id === 'starter' ? p.defaultTc : nt.tc && id !== 'expert' })
              return (
                <button
                  key={id}
                  onClick={() =>
                    setNt({ ...nt, preset: id, alpha: p.defaultAlpha, tc: p.defaultTc })
                  }
                  className={`w-full rounded-xl border p-4 text-left transition-colors ${
                    nt.preset === id
                      ? 'border-gold-500/50 bg-gold-500/[0.06]'
                      : 'border-white/10 hover:border-white/20'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-zinc-100">{p.name}</span>
                    <span className="font-mono text-xs tabular-nums text-zinc-500">{formatBytes(bytes)}</span>
                  </div>
                  <div className="mt-1 text-xs leading-relaxed text-zinc-400">{p.tagline}</div>
                  {id === 'expert' && (
                    <div className="mt-1.5 text-[11px] font-medium text-amber-400/80">
                      ⚠ 268 MB of weights — needs a beefy machine, checkpoints take a moment
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        )}

        {step === 'arch' && kind === 'dqn' && (
          <div className="space-y-3">
            <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">Hidden layers</div>
            {HIDDEN_PRESETS.map((h) => (
              <button
                key={h.label}
                onClick={() => setDqn({ ...dqn, hidden: h.layers })}
                className={`w-full rounded-xl border p-3.5 text-left text-sm transition-colors ${
                  dqn.hidden.join() === h.layers.join()
                    ? 'border-gold-500/50 bg-gold-500/[0.06] text-zinc-100'
                    : 'border-white/10 text-zinc-300 hover:border-white/20'
                }`}
              >
                {h.label}
              </button>
            ))}
            <div className="pt-1 text-xs font-medium uppercase tracking-wide text-zinc-500">Reward shaping</div>
            <div className="flex gap-2">
              {(['scaled', 'log'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setDqn({ ...dqn, shaping: { ...dqn.shaping, mode: m } })}
                  className={`rounded-lg border px-3 py-1.5 text-sm transition-colors ${
                    dqn.shaping.mode === m
                      ? 'border-gold-500/50 bg-gold-500/[0.06] text-zinc-100'
                      : 'border-white/10 text-zinc-400'
                  }`}
                >
                  {m === 'scaled' ? 'score ÷ 1024' : 'log₂ score'}
                </button>
              ))}
            </div>
            <Check
              label="Survival bonus (+0.01 per move)"
              checked={dqn.shaping.survivalBonus}
              onChange={(v) => setDqn({ ...dqn, shaping: { ...dqn.shaping, survivalBonus: v } })}
            />
            <Check
              label="Game-over penalty (−1)"
              checked={dqn.shaping.terminalPenalty}
              onChange={(v) => setDqn({ ...dqn, shaping: { ...dqn.shaping, terminalPenalty: v } })}
            />
            <p className="text-[11px] leading-relaxed text-zinc-600">
              Shaping changes only the reward signal the net learns from — never the move choice.
            </p>
          </div>
        )}

        {step === 'tune' && kind === 'ntuple' && (
          <div className="space-y-4">
            <Num
              label="Learning rate α"
              hint="Total step size per update, split across active weights. With TC on, 1.0 is the classic setting."
              value={nt.alpha}
              step={0.05}
              min={0.001}
              max={2}
              onChange={(v) => setNt({ ...nt, alpha: v })}
            />
            <Check
              label="Temporal Coherence (self-tuning per-weight learning rates)"
              hint={nt.preset !== 'starter' ? 'Triples memory on this preset' : 'Recommended — fast and stable'}
              checked={nt.tc}
              onChange={(v) => setNt({ ...nt, tc: v })}
            />
            <Num
              label="Optimistic initialization V₀"
              hint="Start every position looking valuable (try 80,000) — drives systematic exploration. 0 = off."
              value={nt.optimisticInit}
              step={10000}
              min={0}
              max={500000}
              onChange={(v) => setNt({ ...nt, optimisticInit: v })}
            />
            <Num
              label="Planning depth (inference)"
              hint="1 = pure learned policy. 2–3 = expectimax search over spawns using ONLY the learned values — stronger play, slower moves."
              value={nt.planningDepth}
              step={1}
              min={1}
              max={3}
              onChange={(v) => setNt({ ...nt, planningDepth: v })}
            />
            <Num
              label="Grade strength κ"
              hint="How hard a manual Good/Bad grade pushes the value function."
              value={nt.gradeStrength}
              step={0.5}
              min={0.5}
              max={5}
              onChange={(v) => setNt({ ...nt, gradeStrength: v })}
            />
          </div>
        )}

        {step === 'tune' && kind === 'dqn' && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-4">
            <Num label="Learning rate" value={dqn.lr} step={0.0001} min={0.00001} max={0.01} onChange={(v) => setDqn({ ...dqn, lr: v })} hint="Adam step size" />
            <Num label="Discount γ" value={dqn.gamma} step={0.005} min={0.8} max={0.999} onChange={(v) => setDqn({ ...dqn, gamma: v })} hint="How much the future matters" />
            <Num label="ε start" value={dqn.epsStart} step={0.05} min={0} max={1} onChange={(v) => setDqn({ ...dqn, epsStart: v })} hint="Initial exploration rate" />
            <Num label="ε end" value={dqn.epsEnd} step={0.01} min={0} max={0.5} onChange={(v) => setDqn({ ...dqn, epsEnd: v })} hint="Long-run exploration floor" />
            <Num label="ε decay (moves)" value={dqn.epsDecayMoves} step={5000} min={1000} max={1000000} onChange={(v) => setDqn({ ...dqn, epsDecayMoves: v })} hint="Time constant of the decay" />
            <Num label="Replay size" value={dqn.replaySize} step={5000} min={2000} max={200000} onChange={(v) => setDqn({ ...dqn, replaySize: v })} hint="Experience memory" />
            <Num label="Batch size" value={dqn.batchSize} step={16} min={16} max={256} onChange={(v) => setDqn({ ...dqn, batchSize: v })} hint="Samples per learn step" />
            <Num label="Train every N moves" value={dqn.trainFreq} step={1} min={1} max={16} onChange={(v) => setDqn({ ...dqn, trainFreq: v })} hint="Learn-step frequency" />
            <Num label="Target sync (steps)" value={dqn.targetSync} step={500} min={100} max={20000} onChange={(v) => setDqn({ ...dqn, targetSync: v })} hint="Stability trick: frozen twin network" />
            <Num label="Grade strength κ" value={dqn.gradeStrength} step={0.5} min={0.5} max={5} onChange={(v) => setDqn({ ...dqn, gradeStrength: v })} hint="Impact of manual grades" />
          </div>
        )}

        {step === 'review' && (
          <div className="space-y-4">
            <label className="block">
              <span className="text-sm font-medium text-zinc-300">Model name</span>
              <input
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && create()}
                placeholder={kind === 'ntuple' ? 'e.g. Goldie' : 'e.g. Slowpoke'}
                className="mt-1.5 w-full rounded-lg border border-white/10 bg-surface-2 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-gold-500/50 focus:outline-none"
              />
            </label>
            <div className="rounded-xl border border-white/5 bg-surface-2 p-4 font-mono text-xs leading-relaxed text-zinc-400">
              {kind === 'ntuple' ? (
                <>
                  <Row k="Architecture" v={NTUPLE_PRESETS[nt.preset].name} />
                  <Row k="Weights" v={formatCompact(estimateModelBytes({ ...nt, tc: false }) / 4)} />
                  <Row k="Memory" v={formatBytes(memory)} />
                  <Row k="α / TC" v={`${nt.alpha} / ${nt.tc ? 'on' : 'off'}`} />
                  <Row k="Optimistic V₀" v={String(nt.optimisticInit)} />
                  <Row k="Planning" v={nt.planningDepth === 1 ? 'pure policy' : `expectimax ×${nt.planningDepth}`} />
                </>
              ) : (
                <>
                  <Row k="Architecture" v={`256 → ${dqn.hidden.join(' → ')} → 4`} />
                  <Row k="Memory (incl. replay)" v={formatBytes(memory)} />
                  <Row k="lr / γ" v={`${dqn.lr} / ${dqn.gamma}`} />
                  <Row k="ε" v={`${dqn.epsStart} → ${dqn.epsEnd} over ~${formatCompact(dqn.epsDecayMoves * 3)} moves`} />
                  <Row k="Replay / batch" v={`${formatCompact(dqn.replaySize)} / ${dqn.batchSize}`} />
                </>
              )}
            </div>
            <p className="text-xs leading-relaxed text-zinc-600">
              It will play its first games at pure-random level. Expect chaos — then watch the
              learning curve.
            </p>
          </div>
        )}

        <div className="mt-6 flex items-center justify-between">
          <button
            onClick={() => (stepIdx === 0 ? close() : setStep(steps[stepIdx - 1]))}
            className="rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 transition-colors hover:bg-white/5 hover:text-zinc-200"
          >
            {stepIdx === 0 ? 'Cancel' : '← Back'}
          </button>
          {step === 'review' ? (
            <button
              onClick={create}
              className="rounded-lg bg-gold-500 px-4 py-2 text-sm font-semibold text-zinc-950 transition-colors hover:bg-gold-400"
            >
              Create & load
            </button>
          ) : (
            <button
              onClick={() => setStep(steps[stepIdx + 1])}
              className="rounded-lg bg-zinc-700 px-4 py-2 text-sm font-semibold text-zinc-100 transition-colors hover:bg-zinc-600"
            >
              Next →
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function KindCard({
  selected,
  onClick,
  title,
  badge,
  body,
}: {
  selected: boolean
  onClick: () => void
  title: string
  badge: string
  body: string
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full rounded-xl border p-4 text-left transition-colors ${
        selected ? 'border-gold-500/50 bg-gold-500/[0.06]' : 'border-white/10 hover:border-white/20'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-zinc-100">{title}</span>
        <span className="rounded bg-white/5 px-1.5 py-0.5 text-[11px] font-medium text-zinc-400">
          {badge}
        </span>
      </div>
      <div className="mt-1 text-xs leading-relaxed text-zinc-400">{body}</div>
    </button>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 py-0.5">
      <span className="text-zinc-600">{k}</span>
      <span className="text-zinc-300">{v}</span>
    </div>
  )
}

function Num({
  label,
  hint,
  value,
  step,
  min,
  max,
  onChange,
}: {
  label: string
  hint?: string
  value: number
  step: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-zinc-300">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const v = Number(e.target.value)
          if (Number.isFinite(v)) onChange(v)
        }}
        className="mt-1 w-full rounded-lg border border-white/10 bg-surface-2 px-3 py-1.5 font-mono text-sm tabular-nums text-zinc-100 focus:border-gold-500/50 focus:outline-none"
      />
      {hint && <span className="mt-0.5 block text-[11px] leading-relaxed text-zinc-600">{hint}</span>}
    </label>
  )
}

function Check({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint?: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 accent-gold-500"
      />
      <span>
        <span className="block text-sm text-zinc-300">{label}</span>
        {hint && <span className="block text-[11px] text-zinc-600">{hint}</span>}
      </span>
    </label>
  )
}
