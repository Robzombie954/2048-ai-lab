import { useMemo } from 'react'
import type uPlot from 'uplot'
import { useLabStore } from '../state/labStore'
import { formatCompact, formatInt } from '../lib/format'
import { AXIS_STYLE, UPlotChart } from './charts/UPlotChart'

export function TrajectoryPanel() {
  const activeDoc = useLabStore((s) => s.activeDoc)
  const mode = useLabStore((s) => s.mode)
  const points = useLabStore((s) => s.trajectory)
  const summary = useLabStore((s) => s.gameOverSummary)

  const visible = activeDoc && (mode !== 'turbo' || summary)
  const data = useMemo<uPlot.AlignedData>(() => {
    const safe = points.length ? points : [{ move: 0, score: 0 }]
    return [safe.map((p) => p.move), safe.map((p) => p.score)]
  }, [points])
  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(
    () => ({
      legend: { show: false },
      cursor: { points: { show: false } },
      scales: { x: { time: false } },
      axes: [
        { ...AXIS_STYLE, size: 28 },
        { ...AXIS_STYLE, size: 48 },
      ],
      series: [
        {},
        { stroke: '#f59e0b', width: 2, fill: 'rgba(245,158,11,0.08)', points: { show: false } },
      ],
    }),
    [],
  )

  if (!visible) return null

  return (
    <div className="w-full max-w-md rounded-xl border border-white/5 bg-surface-1 p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
            Current game trajectory
          </div>
          <div className="text-[11px] text-zinc-600">score by move, kept after single-game runs</div>
        </div>
        {summary && (
          <div className="rounded-lg border border-gold-500/20 bg-gold-500/[0.06] px-2.5 py-1.5 text-right">
            <div className="text-[10px] uppercase tracking-wide text-gold-300/80">Game over</div>
            <div className="font-mono text-sm font-semibold tabular-nums text-gold-200">
              {formatInt(summary.score)} · {formatInt(summary.moves)} moves
            </div>
          </div>
        )}
      </div>
      <UPlotChart options={options} data={data} height={120} />
      <div className="mt-2 flex justify-between text-[11px] text-zinc-600">
        <span>{formatCompact(points[points.length - 1]?.move ?? 0)} moves plotted</span>
        <span>final max tile stays on board after one-game mode</span>
      </div>
    </div>
  )
}
