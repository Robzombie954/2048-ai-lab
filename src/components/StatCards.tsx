import { useMemo } from 'react'
import { useLabStore } from '../state/labStore'
import { formatCompact, formatDuration, formatInt } from '../lib/format'
import { normalizeBuckets } from '../stats/chartSeries'

export function StatCards() {
  const live = useLabStore((s) => s.live)
  const activeDoc = useLabStore((s) => s.activeDoc)
  const buckets = useLabStore((s) => s.buckets)
  const partial = useLabStore((s) => s.partialBucket)
  const recentMoves = useMemo(() => {
    const all = normalizeBuckets(partial ? [...buckets, partial] : buckets)
    const recent = all.slice(-5)
    const n = recent.reduce((sum, b) => sum + b.n, 0)
    const moves = recent.reduce((sum, b) => sum + b.movesSum, 0)
    return n > 0 ? moves / n : 0
  }, [buckets, partial])

  if (!activeDoc) return null

  const games = live?.games ?? activeDoc.games
  const totalMoves = live?.totalMoves ?? activeDoc.moves
  const lifetimeMovesPerGame = games > 0 ? totalMoves / games : 0

  const cards: { label: string; value: string; sub?: string }[] = [
    { label: 'Games played', value: formatCompact(games) },
    {
      label: 'Moves / sec',
      value: live && live.movesPerSec > 0 ? formatCompact(live.movesPerSec) : '—',
      sub: `${formatCompact(totalMoves)} total moves`,
    },
    {
      label: 'Games / sec',
      value: live && (live.gamesPerSec ?? 0) > 0 ? formatCompact(live.gamesPerSec!) : '—',
      sub: live && (live.gamesPerSec ?? 0) > 0 ? 'training throughput' : undefined,
    },
    {
      label: 'Avg moves/game',
      value: lifetimeMovesPerGame > 0 ? formatCompact(lifetimeMovesPerGame) : '—',
      sub: recentMoves > 0 ? `${formatCompact(recentMoves)} recent` : 'lifetime survival length',
    },
    {
      label: 'Avg score (last 100)',
      value: formatCompact(live?.recentAvg ?? 0),
      sub: 'random baseline ≈ 1,090',
    },
    { label: 'Best score', value: formatCompact(live?.bestScore ?? activeDoc.bestScore) },
    {
      label: 'Training time',
      value: formatDuration(live?.trainMs ?? activeDoc.trainMs),
    },
    {
      label: 'Parameters',
      value: formatCompact(live?.paramCount ?? 0),
      sub: live ? `${(live.memoryBytes / (1 << 20)).toFixed(1)} MB` : undefined,
    },
  ]

  return (
    <div>
      <div className="grid grid-cols-2 gap-3">
        {cards.map((c) => (
          <div key={c.label} className="rounded-xl border border-white/5 bg-surface-1 p-4">
            <div className="text-xs font-medium text-zinc-500">{c.label}</div>
            <div className="mt-1 font-mono text-2xl font-semibold tabular-nums text-zinc-100">
              {c.value}
            </div>
            {c.sub && <div className="mt-0.5 text-[11px] text-zinc-600">{c.sub}</div>}
          </div>
        ))}
      </div>
      {live && (
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 rounded-xl border border-white/5 bg-surface-1 px-4 py-3 font-mono text-[11px] tabular-nums text-zinc-500">
          <span title="Mean absolute TD error (exponential moving average)">
            |δ| {live.meanAbsTdError.toFixed(live.meanAbsTdError >= 10 ? 0 : 3)}
          </span>
          <span>α {live.learningRate}</span>
          {live.epsilon !== undefined && <span>ε {live.epsilon.toFixed(3)}</span>}
          {live.replayFill !== undefined && (
            <span>replay {(live.replayFill * 100).toFixed(0)}%</span>
          )}
          <span>game #{formatInt(live.games + 1)}</span>
        </div>
      )}
    </div>
  )
}

