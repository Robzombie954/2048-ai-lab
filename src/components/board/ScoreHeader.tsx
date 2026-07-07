import { useLabStore } from '../../state/labStore'
import { bridge } from '../../state/workerBridge'
import { formatInt } from '../../lib/format'
import { tileTheme } from '../../lib/tileTheme'

export function ScoreHeader() {
  const score = useLabStore((s) => s.score)
  const gameIndex = useLabStore((s) => s.gameIndex)
  const live = useLabStore((s) => s.live)
  const activeDoc = useLabStore((s) => s.activeDoc)
  const highScores = useLabStore((s) => s.highScoreGames)

  const bestScore = Math.max(live?.bestScore ?? 0, activeDoc?.bestScore ?? 0, score)
  const bestExp = Math.max(live?.bestExp ?? 0, activeDoc?.bestExp ?? 0)
  const replayCount = highScores?.length ?? 0

  return (
    <div className="flex w-full max-w-md items-end justify-between">
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
          Game {formatInt(gameIndex + 1)}
        </div>
        <div className="font-mono text-3xl font-semibold tabular-nums text-zinc-100">
          {formatInt(score)}
        </div>
      </div>
      <div className="flex items-center gap-4">
        {activeDoc && (
          <button
            onClick={() => bridge.newGame()}
            title="Deal a fresh game and watch the model play it from scratch (press N)"
            className="flex h-8 items-center gap-1.5 rounded-lg border border-white/10 px-2.5 text-xs font-medium text-zinc-400 transition-colors hover:border-white/20 hover:bg-white/5 hover:text-zinc-200"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="size-3.5"
            >
              <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
            </svg>
            New game
          </button>
        )}
        <div className="text-right">
          <div className="text-xs font-medium uppercase tracking-wide text-zinc-500 flex items-center gap-1">
            Best
            {replayCount > 0 && (
              <span 
                className="rounded bg-gold-500/20 px-1 py-px text-[9px] font-mono text-gold-400 cursor-help" 
                title={`${replayCount} replayable high-score game(s) — see below or in sidebar`}
              >
                {replayCount} replay{replayCount === 1 ? '' : 's'}
              </span>
            )}
          </div>
          <div className="font-mono text-lg font-medium tabular-nums text-zinc-300">
            {formatInt(bestScore)}
          </div>
        </div>
        {bestExp > 0 && (
          <div
            className="flex h-11 w-11 items-center justify-center rounded-lg text-xs font-bold"
            style={{
              background: tileTheme(bestExp).bg,
              color: tileTheme(bestExp).fg,
              border: tileTheme(bestExp).border,
            }}
            title={`Best tile ever: ${1 << bestExp}`}
          >
            {formatInt(1 << bestExp)}
          </div>
        )}
      </div>
    </div>
  )
}
