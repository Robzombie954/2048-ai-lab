import { useEffect, useState } from 'react'
import { useLabStore, setLab } from './state/labStore'
import { bridge } from './state/workerBridge'
import { BoardStage } from './components/board/BoardStage'
import { ScoreHeader } from './components/board/ScoreHeader'
import { DirectionViz } from './components/viz/DirectionViz'
import { TrainingControls } from './components/controls/TrainingControls'
import { ModelLibrary } from './components/library/ModelLibrary'
import { NewModelWizard } from './components/wizard/NewModelWizard'
import { ChartsPanel } from './components/charts/ChartsPanel'
import { StatCards } from './components/StatCards'
import { MilestoneRibbon } from './components/MilestoneRibbon'
import { TrajectoryPanel } from './components/TrajectoryPanel'
import { GpuBenchmarkPanel } from './components/GpuBenchmarkPanel'
import { formatAgo, formatCompact, formatInt } from './lib/format'

export default function App() {
  const activeDoc = useLabStore((s) => s.activeDoc)
  const running = useLabStore((s) => s.running)
  const mode = useLabStore((s) => s.mode)
  const highScores = useLabStore((s) => s.highScoreGames)
  const replay = useLabStore((s) => s.replay)
  const [mobileLibOpen, setMobileLibOpen] = useState(false)
  // Auto-close the mobile library drawer once a model is picked/created.
  useEffect(() => {
    setMobileLibOpen(false)
  }, [activeDoc?.id])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-white/5 px-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMobileLibOpen(true)}
            className="rounded-md border border-white/10 px-2 py-1 text-xs font-medium text-zinc-300 transition-colors hover:bg-white/5 md:hidden"
            aria-label="Open model library"
          >
            ☰ Models
          </button>
          <h1 className="text-sm font-bold tracking-tight text-zinc-100">
            2048 <span className="text-gold-500">AI Lab</span>
          </h1>
          <span className="hidden text-xs text-zinc-600 sm:block">
            real reinforcement learning, live in your browser
          </span>
        </div>
        {activeDoc && (
          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium text-zinc-300">{activeDoc.name}</span>
            <span
              className={`rounded-full px-2 py-0.5 font-medium ${
                running
                  ? 'bg-emerald-500/15 text-emerald-300'
                  : 'bg-zinc-800 text-zinc-500'
              }`}
            >
              {running ? `training · ${mode}` : 'paused'}
            </span>
          </div>
        )}
      </header>

      <ResumeBanner />

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[260px_minmax(0,1fr)] xl:grid-cols-[280px_minmax(0,1fr)_400px]">
        <aside className="hidden border-r border-white/5 md:flex md:h-full md:flex-col md:overflow-hidden">
          <div className="flex-1 min-h-0">
            <ModelLibrary />
          </div>
          {activeDoc && (
            <div className="flex-shrink-0 border-t border-white/5 p-3 text-xs bg-surface-0/50">
              <div className="mb-1.5 flex items-center justify-between text-[10px] font-semibold uppercase tracking-widest text-zinc-400">
                <span>Top Score Replays</span>
                <span className="font-mono text-zinc-500">{highScores.length}</span>
              </div>
              {highScores.length >= 2 && (
                <button
                  onClick={() => bridge.playAllHighScores(highScores)}
                  className="mb-1.5 w-full rounded bg-gold-500/10 px-2 py-1 text-[10px] font-medium text-gold-300 hover:bg-gold-500/20 active:bg-gold-500/30"
                >
                  ▶ Play all (lowest → highest)
                </button>
              )}
              {highScores.length > 0 ? (
                <div className="max-h-[160px] space-y-1 overflow-y-auto pr-1 text-[11px]">
                  {[...highScores].reverse().slice(0, 8).map((r, i) => (
                    <div key={i} className="flex items-center justify-between gap-2 rounded border border-white/10 bg-surface-2 px-2 py-1 hover:border-white/20 hover:bg-surface-3">
                      <div>
                        <div className="font-mono text-gold-300">g{formatInt(r.summary.game + 1)} · {formatCompact(r.summary.score)}</div>
                        <div className="text-[10px] text-zinc-400">tile {(1 << r.summary.maxExp) || 0} · {r.summary.moves} mv</div>
                      </div>
                      <button
                        onClick={() => bridge.startReplay(r)}
                        className="rounded bg-gold-500/90 px-2 py-0.5 text-[10px] font-semibold text-zinc-950 hover:bg-gold-400 active:bg-gold-500"
                        title="Replay this high-score game using the saved seed + move recipe"
                      >
                        Replay
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-[10px] text-zinc-400 py-1">No replayable records yet. New games that beat the current best will appear here with full replays.</div>
              )}
              {highScores.length > 8 && (
                <div className="mt-1 text-[10px] text-zinc-500">+{highScores.length - 8} more in the list below</div>
              )}
            </div>
          )}
        </aside>

        <main className="overflow-y-scroll [scrollbar-gutter:stable]">
          <div className="mx-auto flex max-w-2xl flex-col items-center gap-4 p-4 sm:p-6">
            <ScoreHeader />
            <BoardStage />
            <DirectionViz />
            <TrainingControls />

            {/* High-score game replay viewer */}
            {replay && (
              <div className="w-full max-w-md rounded-xl border border-gold-500/30 bg-surface-1 p-4">
                <div className="mb-2 flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-gold-300">
                      Replaying historical high score
                      {replay.playlistTotal && replay.playlistCurrent && (
                        <span className="ml-2 text-[10px] text-gold-400/80">({replay.playlistCurrent} / {replay.playlistTotal})</span>
                      )}
                    </div>
                    <div className="font-mono text-xs text-zinc-400">
                      Game #{formatInt(replay.record.summary.game + 1)} — score {formatCompact(replay.record.summary.score)} — {replay.record.summary.moves} moves
                    </div>
                  </div>
                  <button
                    onClick={() => bridge.stopReplay()}
                    className="rounded-md border border-white/10 px-2 py-0.5 text-xs hover:bg-white/5"
                  >
                    {replay.playlistTotal ? 'Stop sequence' : 'Exit replay'}
                  </button>
                </div>

                <div className="mb-2 flex items-center gap-2 text-xs">
                  <button
                    onClick={() => bridge.toggleReplayPlay()}
                    className="rounded-lg bg-gold-500 px-3 py-1 text-sm font-semibold text-zinc-950 hover:bg-gold-400"
                  >
                    {replay.playing ? '⏸ Pause' : '▶ Play'}
                  </button>
                  <button
                    onClick={() => bridge.replayStep()}
                    disabled={replay.index >= replay.record.actions.length}
                    className="rounded-md border border-white/10 px-2 py-1 disabled:opacity-40"
                  >
                    Step
                  </button>
                  <button
                    onClick={() => {
                      // jump to start
                      bridge.stopReplay()
                      bridge.startReplay(replay.record)
                    }}
                    className="rounded-md border border-white/10 px-2 py-1 text-xs"
                  >
                    Restart
                  </button>

                  <div className="ml-auto flex items-center gap-2 text-zinc-400">
                    <span>Speed</span>
                    <input
                      type="range"
                      min={1}
                      max={25}
                      value={replay.speed}
                      onChange={(e) => bridge.setReplaySpeed(Number(e.target.value))}
                      className="w-20 accent-gold-500"
                    />
                    <span className="w-8 font-mono text-right tabular-nums">{replay.speed}</span>
                  </div>
                </div>

                <div className="h-1.5 w-full overflow-hidden rounded bg-white/10">
                  <div
                    className="h-1.5 bg-gold-500 transition-all"
                    style={{ width: `${Math.min(100, (replay.index / Math.max(1, replay.record.actions.length)) * 100)}%` }}
                  />
                </div>
                <div className="mt-1 flex justify-between text-[10px] text-zinc-500 font-mono">
                  <span>move {formatInt(replay.index)} / {formatInt(replay.record.actions.length)}</span>
                  <span>{replay.index >= replay.record.actions.length ? 'Done — final board shown' : 'replaying from seed + actions'}</span>
                </div>
                <div className="mt-1 text-[10px] text-zinc-400">
                  Previous best was {formatCompact(replay.record.previousBestScore)} → this run set the new record.
                </div>
              </div>
            )}

            <GpuBenchmarkPanel />
            <TrajectoryPanel />

            {/* Top Score Replays / history (always visible for active model) */}
            {activeDoc && !replay && (
              <div className="w-full max-w-md rounded-xl border border-white/5 bg-surface-1 p-3 text-xs pb-4">
                <div className="mb-1 flex items-center justify-between">
                  <div className="font-semibold text-zinc-300 flex items-center gap-2">
                    Top Score Replays
                    {highScores.length > 0 && <span className="text-[10px] text-zinc-500 font-normal">({highScores.length})</span>}
                  </div>
                  {highScores.length >= 2 && (
                    <button
                      onClick={() => bridge.playAllHighScores(highScores)}
                      className="rounded bg-gold-500/10 px-2 py-0.5 text-[10px] text-gold-300 hover:bg-gold-500/20"
                    >
                      ▶ Play all (lowest → highest)
                    </button>
                  )}
                </div>
                {highScores.length > 0 ? (
                  <div className="max-h-[220px] overflow-y-auto divide-y divide-white/10 text-[11px] -mx-1 px-1">
                    {[...highScores].sort((a, b) => b.summary.score - a.summary.score).map((r) => (
                      <div key={`${r.modelId}-${r.summary.game}`} className="flex items-center justify-between py-1.5 px-1 rounded hover:bg-surface-2">
                        <div className="font-mono">
                          g{formatInt(r.summary.game + 1)}: <span className="text-gold-300">{formatCompact(r.summary.score)}</span> (tile {formatInt((1 << r.summary.maxExp) || 0)})
                        </div>
                        <button 
                          onClick={() => bridge.startReplay(r)} 
                          className="rounded bg-gold-500/10 px-2 py-0.5 text-[10px] text-gold-300 hover:bg-gold-500/20 active:bg-gold-500/30"
                        >
                          Replay
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-2 text-[11px] text-zinc-400">
                    No replayable high-score games saved yet for this model.<br />
                    Train until the model sets a new best score — the full game (seed + every move) will be recorded here so you can replay exactly how it achieved the new top score.
                  </div>
                )}
                <div className="mt-1 text-[10px] text-zinc-500">Only games that raised the all-time best score for the model are saved with replay recipes.</div>
              </div>
            )}

            <div className="w-full space-y-4 xl:hidden">
              <StatCards />
              <ChartsPanel />
            </div>
          </div>
        </main>

        <aside className="hidden overflow-y-scroll border-l border-white/5 [scrollbar-gutter:stable] xl:block">
          <div className="space-y-4 p-4">
            <StatCards />
            <ChartsPanel />
          </div>
        </aside>
      </div>

      {mobileLibOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setMobileLibOpen(false)}
          />
          <div className="flash-in absolute left-0 top-0 flex h-full w-[280px] max-w-[85vw] flex-col border-r border-white/10 bg-surface-1">
            <div className="flex items-center justify-end border-b border-white/5 px-2 py-1.5">
              <button
                onClick={() => setMobileLibOpen(false)}
                className="rounded-md px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-white/5"
              >
                ✕ Close
              </button>
            </div>
            <div className="min-h-0 flex-1">
              <ModelLibrary />
            </div>
          </div>
        </div>
      )}

      <MilestoneRibbon />
      <NewModelWizard />
      <Toast />
    </div>
  )
}

function ResumeBanner() {
  const hint = useLabStore((s) => s.resumeHint)
  if (!hint) return null
  return (
    <div className="flash-in flex items-center justify-between gap-4 border-b border-gold-500/20 bg-gold-500/[0.07] px-4 py-2.5">
      <span className="text-sm text-zinc-200">
        <span className="font-semibold">{hint.name}</span> was training in {hint.mode} mode when
        the tab closed ({formatAgo(Date.now() - hint.ageMs)}) — pick up where it left off?
      </span>
      <div className="flex shrink-0 gap-2">
        <button
          onClick={() => bridge.resume()}
          className="rounded-lg bg-gold-500 px-3 py-1.5 text-xs font-semibold text-zinc-950 transition-colors hover:bg-gold-400"
        >
          Resume training
        </button>
        <button
          onClick={() => bridge.dismissResume()}
          className="rounded-lg px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:bg-white/5"
        >
          Just load it
        </button>
      </div>
    </div>
  )
}

function Toast() {
  const toast = useLabStore((s) => s.toast)
  const errorMsg = useLabStore((s) => s.errorMsg)
  if (!toast && !errorMsg) return null
  return (
    <div className="pointer-events-none fixed bottom-14 right-4 z-50 space-y-2">
      {toast && (
        <div className="flash-in rounded-lg border border-white/10 bg-surface-3 px-3.5 py-2.5 text-sm text-zinc-200 shadow-lg">
          {toast}
        </div>
      )}
      {errorMsg && (
        <div className="pointer-events-auto flash-in flex items-center gap-3 rounded-lg border border-red-500/30 bg-red-950/80 px-3.5 py-2.5 text-sm text-red-200 shadow-lg">
          <span>{errorMsg}</span>
          <button onClick={() => setLab({ errorMsg: null })} className="text-red-400 hover:text-red-200">
            ✕
          </button>
        </div>
      )}
    </div>
  )
}


