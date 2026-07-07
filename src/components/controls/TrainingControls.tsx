import { useEffect, useState } from 'react'
import { useLabStore } from '../../state/labStore'
import { bridge } from '../../state/workerBridge'
import type { TrainingMode } from '../../shared/types'
import { formatAgo, formatBytes, formatCompact } from '../../lib/format'

const MODES: { id: TrainingMode; label: string; hint: string }[] = [
  { id: 'watch', label: 'Watch', hint: 'Animated self-play at viewing speed' },
  { id: 'turbo', label: 'Turbo', hint: 'Max-speed training in the background thread' },
  { id: 'grade', label: 'Grade', hint: 'You grade every move: 1 good · 2 neutral · 3 bad' },
]

export function TrainingControls() {
  const running = useLabStore((s) => s.running)
  const mode = useLabStore((s) => s.mode)
  const mps = useLabStore((s) => s.movesPerSec)
  const depth = useLabStore((s) => s.planningDepth)
  const activeDoc = useLabStore((s) => s.activeDoc)
  const awaitingGrade = useLabStore((s) => s.awaitingGrade)
  const oneGameActive = useLabStore((s) => s.oneGameActive)
  const lastCheckpointAt = useLabStore((s) => s.lastCheckpointAt)
  const lastCheckpointBytes = useLabStore((s) => s.lastCheckpointBytes)
  const replay = useLabStore((s) => s.replay)

  const [batchSize, setBatchSize] = useState(100)
  const targetCount = useLabStore((s) => s.targetGameCount)
  const liveGames = useLabStore((s) => s.live?.games)
  const activeGames = useLabStore((s) => s.activeDoc?.games ?? 0)

  useEffect(() => {
    if (mode !== 'grade') return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '1') bridge.grade(1)
      else if (e.key === '2') bridge.grade(0)
      else if (e.key === '3') bridge.grade(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return
      if (e.key === 'n' || e.key === 'N') bridge.newGame()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!activeDoc) return null
  const isNTuple = activeDoc.kind === 'ntuple'
  const annealOn = activeDoc.config.kind === 'ntuple' && !!activeDoc.config.anneal

  return (
    <div className="w-full max-w-md space-y-3">
      <div className="flex items-center gap-2">
        <button
          onClick={() => (running ? bridge.stop() : bridge.start())}
          className={`h-10 flex-1 rounded-lg font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gold-400 ${
            running
              ? 'bg-zinc-800 text-zinc-200 hover:bg-zinc-700'
              : 'bg-gold-500 text-zinc-950 hover:bg-gold-400'
          }`}
        >
          {running ? '■ Pause training' : '▶ Train'}
        </button>
        <div className="flex rounded-lg border border-white/10 p-0.5">
          {MODES.map((m) => (
            <button
              key={m.id}
              title={m.hint}
              onClick={() => bridge.setMode(m.id)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                mode === m.id
                  ? 'bg-zinc-700 text-zinc-100'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {mode === 'watch' && (
        <label className="flex items-center gap-3 text-sm text-zinc-400">
          <span className="w-24 shrink-0">Speed</span>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={mps}
            onChange={(e) => bridge.setSpeed(Number(e.target.value))}
            className="flex-1 accent-gold-500"
          />
          <span className="w-16 text-right font-mono text-xs tabular-nums">{mps} mv/s</span>
        </label>
      )}

      {isNTuple && (
        <div className="flex items-center gap-3 text-sm text-zinc-400">
          <span className="w-24 shrink-0">Planning</span>
          <div className="flex items-center gap-1">
            {[1, 2, 3].map((d) => (
              <button
                key={d}
                onClick={() => bridge.setPlanningDepth(d)}
                className={`h-7 w-7 rounded-md text-xs font-medium transition-colors ${
                  depth === d
                    ? 'bg-zinc-700 text-zinc-100'
                    : 'bg-white/[0.04] text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {d}
              </button>
            ))}
          </div>
          <span
            className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
              depth === 1 ? 'bg-emerald-500/10 text-emerald-300' : 'bg-gold-500/10 text-gold-300'
            }`}
            title="Planning searches future spawns but only ever uses the learned value function"
          >
            {depth === 1 ? 'pure policy' : `planning ×${depth}`}
          </span>
        </div>
      )}

      {isNTuple && (
        <div className="flex items-center gap-3 text-sm text-zinc-400">
          <span className="w-24 shrink-0">Anneal α</span>
          <button
            role="switch"
            aria-checked={annealOn}
            onClick={() => bridge.setAnnealing(!annealOn)}
            title="Gradually lower the learning rate as games accumulate — quiets the plateau noise once a model has mostly converged"
            className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${
              annealOn ? 'bg-gold-500' : 'bg-white/10'
            }`}
          >
            <span
              className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                annealOn ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </button>
          <span className="text-[11px] text-zinc-500">
            {annealOn ? 'learning rate decays to settle the plateau' : 'constant learning rate'}
          </span>
        </div>
      )}

      {mode === 'grade' && (
        <div
          className={`rounded-xl border p-3 transition-colors ${
            awaitingGrade ? 'border-gold-500/40 bg-gold-500/5' : 'border-white/5 bg-surface-1'
          }`}
        >
          <div className="mb-2 text-xs text-zinc-400">
            {awaitingGrade
              ? 'Grade this move — it updates the model for real'
              : running
                ? 'Waiting for the next move…'
                : 'Press Train to start grading moves'}
          </div>
          <div className="grid grid-cols-3 gap-2">
            <button
              disabled={!awaitingGrade}
              onClick={() => bridge.grade(1)}
              className="h-9 rounded-lg bg-emerald-500/15 text-sm font-semibold text-emerald-300 transition-colors hover:bg-emerald-500/25 disabled:opacity-40"
            >
              Good <span className="font-normal text-emerald-500/70">1</span>
            </button>
            <button
              disabled={!awaitingGrade}
              onClick={() => bridge.grade(0)}
              className="h-9 rounded-lg bg-zinc-700/40 text-sm font-semibold text-zinc-300 transition-colors hover:bg-zinc-700/70 disabled:opacity-40"
            >
              Neutral <span className="font-normal text-zinc-500">2</span>
            </button>
            <button
              disabled={!awaitingGrade}
              onClick={() => bridge.grade(-1)}
              className="h-9 rounded-lg bg-red-500/15 text-sm font-semibold text-red-300 transition-colors hover:bg-red-500/25 disabled:opacity-40"
            >
              Bad <span className="font-normal text-red-500/70">3</span>
            </button>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-zinc-600">
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => bridge.playOneGame()}
            className={`rounded-md px-2 py-1 transition-colors hover:bg-white/5 ${
              oneGameActive ? 'text-gold-300' : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Start a fresh game, watch exactly one full game, then stop on the final board"
          >
            {oneGameActive ? 'One game running' : 'Play one game'}
          </button>
          <button
            onClick={() => bridge.saveCheckpoint()}
            className="rounded-md px-2 py-1 text-zinc-400 transition-colors hover:bg-white/5 hover:text-zinc-200"
          >
            Save now
          </button>
          <button
            onClick={() => bridge.exportModel()}
            className="rounded-md px-2 py-1 text-zinc-400 transition-colors hover:bg-white/5 hover:text-zinc-200"
          >
            Export file
          </button>
        </div>
        {lastCheckpointAt && (
          <span title={formatBytes(lastCheckpointBytes)}>
            saved {formatAgo(lastCheckpointAt)}
          </span>
        )}
      </div>

      {/* Batch training: exact number of games */}
      <div className="rounded-lg border border-white/10 bg-surface-1/60 p-3">
        <div className="mb-1.5 flex items-center justify-between text-xs">
          <span className="font-medium text-zinc-400">Train exact # of games</span>
          {targetCount && (
            <span className="rounded bg-gold-500/10 px-1.5 py-0.5 font-mono text-[10px] text-gold-300">
              target {formatCompact(targetCount)} (at {formatCompact(liveGames ?? activeGames)} / {formatCompact((liveGames ?? activeGames) + targetCount)})
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={1}
            max={100000}
            value={batchSize}
            onChange={(e) => setBatchSize(Math.max(1, Math.min(100000, Number(e.target.value) || 1)))}
            disabled={!!replay || running}
            className="w-24 rounded-md border border-white/10 bg-zinc-950 px-2 py-1 font-mono text-sm text-zinc-100 disabled:opacity-50"
          />
          <button
            onClick={() => bridge.trainForGames(batchSize)}
            disabled={running || !!replay}
            className="flex-1 rounded-md bg-zinc-800 px-3 py-1 text-sm font-semibold text-zinc-200 transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            title="Run exactly this many more games in turbo, then auto-pause. Great for before/after comparisons."
          >
            Train {batchSize} games
          </button>
          {targetCount && running && (
            <button onClick={() => bridge.stop()} className="rounded-md px-2 py-1 text-xs text-red-400 hover:bg-red-500/10">Cancel target</button>
          )}
        </div>
        <div className="mt-1 text-[10px] text-zinc-500">Auto-stops after N games. Use with checkpoints or high-score history to compare progress.</div>
      </div>

      {replay && (
        <div className="rounded-lg border border-gold-500/30 bg-gold-500/5 p-2 text-xs text-gold-300">
          Replay active — training paused. Use the replay panel to control playback.
        </div>
      )}
    </div>
  )
}
