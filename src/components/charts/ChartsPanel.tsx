import { useEffect, useMemo, useState } from 'react'
import type uPlot from 'uplot'
import { setLab, useLabStore } from '../../state/labStore'
import { bridge } from '../../state/workerBridge'
import type { StatBucket } from '../../shared/types'
import { formatCompact } from '../../lib/format'
import { tileTheme } from '../../lib/tileTheme'
import { getBuckets } from '../../persistence/statStore'
import { nextPresetUp } from '../../agents/factory'
import {
  bucketStd,
  bucketX,
  finiteOrNull,
  normalizeBuckets,
  projectedTrajectory,
  rollingTrajectory,
  safeBucketMean,
  summarizeTrajectory,
  safeMovesPerGame,
  type TrendLabel,
} from '../../stats/chartSeries'
import { AXIS_STYLE, UPlotChart } from './UPlotChart'

/** Rough expected average-score band each n-tuple preset settles into. */
const PRESET_CEILINGS: Record<string, { lo: number; hi: number; label: string }> = {
  starter: { lo: 25_000, hi: 45_000, label: 'Starter' },
  balanced: { lo: 90_000, hi: 200_000, label: 'Balanced' },
  expert: { lo: 200_000, hi: 500_000, label: 'Expert' },
}

const TREND_STYLE: Record<TrendLabel, { chip: string; text: string }> = {
  rising: { chip: 'border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-200', text: 'rising' },
  converging: { chip: 'border-sky-400/20 bg-sky-400/[0.08] text-sky-200', text: 'converging' },
  converged: {
    chip: 'border-amber-400/20 bg-amber-400/[0.08] text-amber-200',
    text: 'converged · near capacity',
  },
  falling: { chip: 'border-red-400/20 bg-red-400/[0.08] text-red-200', text: 'falling' },
}

/** Step-resample a source series onto a target x-grid (last-known value). */
function resampleOnto(
  targetX: readonly number[],
  srcX: readonly number[],
  srcY: readonly number[],
): (number | null)[] {
  const out: (number | null)[] = []
  let j = 0
  let last: number | null = null
  for (const t of targetX) {
    while (j < srcX.length && srcX[j] <= t) {
      last = Number.isFinite(srcY[j]) ? srcY[j] : last
      j++
    }
    out.push(last)
  }
  return out
}

const RANDOM_BASELINE = 1090

type Tab = 'learning' | 'tiles' | 'tderror' | 'rates' | 'moves'

const TABS: { id: Tab; label: string }[] = [
  { id: 'learning', label: 'Learning curve' },
  { id: 'tiles', label: 'Max tiles' },
  { id: 'tderror', label: 'TD error' },
  { id: 'rates', label: 'ε & α' },
  { id: 'moves', label: 'Moves/game' },
]

function useAllBuckets(): StatBucket[] {
  const buckets = useLabStore((s) => s.buckets)
  const partial = useLabStore((s) => s.partialBucket)
  return useMemo(() => normalizeBuckets(partial ? [...buckets, partial] : buckets), [buckets, partial])
}

export function ChartsPanel() {
  const [tab, setTab] = useState<Tab>('learning')
  const all = useAllBuckets()
  const activeDoc = useLabStore((s) => s.activeDoc)
  if (!activeDoc) return null

  return (
    <div className="rounded-xl border border-white/5 bg-surface-1">
      <div className="flex gap-1 overflow-x-auto border-b border-white/5 p-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`whitespace-nowrap rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              tab === t.id ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="p-3">
        {all.length === 0 ? (
          <div className="flex h-40 flex-col items-center justify-center gap-1 text-center">
            <div className="text-sm font-medium text-zinc-400">No data yet</div>
            <div className="text-xs text-zinc-600">
              Stats appear as the model finishes games — turbo fills this fast
            </div>
          </div>
        ) : tab === 'learning' ? (
          <LearningCurve all={all} />
        ) : tab === 'tiles' ? (
          <TileHistogram all={all} />
        ) : tab === 'tderror' ? (
          <SimpleSeries all={all} label="mean |δ| per game" color="#f0abfc" pick={(b) => b.tdErrAvg} />
        ) : tab === 'rates' ? (
          <RatesChart all={all} />
        ) : (
          <SimpleSeries all={all} label="moves per game" color="#7dd3fc" pick={safeMovesPerGame} />
        )}
      </div>
    </div>
  )
}

const xVals = (all: StatBucket[]) => all.map(bucketX)

function LearningCurve({ all }: { all: StatBucket[] }) {
  const activeDoc = useLabStore((s) => s.activeDoc)
  const models = useLabStore((s) => s.models)
  const planningDepth = useLabStore((s) => s.planningDepth)
  const [yMode, setYMode] = useState<'focus' | 'full'>('focus')
  const [compareId, setCompareId] = useState<string | null>(null)
  const [compareBuckets, setCompareBuckets] = useState<StatBucket[]>([])

  const ceiling =
    activeDoc?.config.kind === 'ntuple' ? PRESET_CEILINGS[activeDoc.config.preset] : null
  const levelUpTo =
    activeDoc?.config.kind === 'ntuple' ? nextPresetUp(activeDoc.config.preset) : null

  useEffect(() => {
    if (!compareId) {
      setCompareBuckets([])
      return
    }
    let alive = true
    void getBuckets(compareId).then((b) => {
      if (alive) setCompareBuckets(normalizeBuckets(b))
    })
    return () => {
      alive = false
    }
  }, [compareId])

  const compareDoc = compareId ? models.find((m) => m.id === compareId) : null
  const compareActive = !!compareId && compareBuckets.length > 0

  const { data, summary, focusMax } = useMemo(() => {
    const x = xVals(all)
    const means = all.map(safeBucketMean)
    const stds = all.map(bucketStd)
    const future = projectedTrajectory(x, means, 14, 8)
    const futureX = future.x.slice(1)
    const combinedX = [...x, ...futureX]
    const pad = (s: (number | null)[]) => [...s, ...futureX.map(() => null)]
    const projection = [...Array(Math.max(0, x.length - 1)).fill(null), ...future.y]

    const bandTop = means.map((m, i) => finiteOrNull(m + stds[i]))
    const bandBottom = means.map((m, i) => finiteOrNull(Math.max(0, m - stds[i])))

    // Focus range: driven by the average ± ~2σ and the expected ceiling — NOT
    // by a single lucky best-game, which is what used to squash the curve.
    let fMax = RANDOM_BASELINE * 3
    for (let i = 0; i < means.length; i++) fMax = Math.max(fMax, means[i] + 2 * stds[i])
    for (const v of future.y) if (v != null) fMax = Math.max(fMax, v)
    if (ceiling) fMax = Math.max(fMax, ceiling.hi)

    const series: uPlot.AlignedData = [
      combinedX,
      pad(all.map((b) => finiteOrNull(b.scoreMax))), // 1 peak (full-range only)
      pad(bandTop), // 2 band top (mean+σ)
      pad(bandBottom), // 3 band bottom (mean−σ)
      pad(means.map(finiteOrNull)), // 4 average (primary)
      pad(rollingTrajectory(x, means, 14)), // 5 trajectory
      projection, // 6 projection
      combinedX.map(() => RANDOM_BASELINE), // 7 random baseline
    ]

    if (compareActive) {
      const cx = compareBuckets.map(bucketX)
      const cMeans = compareBuckets.map(safeBucketMean)
      for (const v of cMeans) fMax = Math.max(fMax, v)
      series.push(resampleOnto(combinedX, cx, cMeans)) // 8 compare average
    }

    return { data: series, summary: summarizeTrajectory(x, means, 14), focusMax: fMax * 1.12 }
  }, [all, ceiling, compareActive, compareBuckets])

  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => {
    const series: uPlot.Series[] = [
      {},
      { stroke: 'rgba(255,255,255,0.14)', width: 1, dash: [2, 3], points: { show: false } },
      { stroke: 'transparent', points: { show: false } },
      { stroke: 'transparent', points: { show: false } },
      { stroke: '#edc22e', width: 2, points: { show: false } },
      { stroke: '#38bdf8', width: 2, points: { show: false } },
      { stroke: '#38bdf8', width: 1.5, dash: [5, 4], points: { show: false } },
      { stroke: '#52525b', width: 1, dash: [4, 4], points: { show: false } },
    ]
    if (compareActive) series.push({ stroke: '#c084fc', width: 2, points: { show: false } })
    return {
      legend: { show: false },
      cursor: { points: { show: false } },
      scales: { x: { time: false }, y: yMode === 'focus' ? { range: [0, focusMax] } : {} },
      axes: [AXIS_STYLE, { ...AXIS_STYLE, size: 52 }],
      bands: [{ series: [2, 3], fill: 'rgba(237,194,46,0.09)' }],
      series,
      hooks: ceiling
        ? {
            draw: [
              (u: uPlot) => {
                const ctx = u.ctx
                const yHi = u.valToPos(ceiling.hi, 'y', true)
                const yLo = u.valToPos(ceiling.lo, 'y', true)
                ctx.save()
                ctx.fillStyle = 'rgba(16,185,129,0.06)'
                ctx.fillRect(u.bbox.left, yHi, u.bbox.width, Math.max(0, yLo - yHi))
                ctx.restore()
              },
            ],
          }
        : undefined,
    }
  }, [yMode, focusMax, ceiling, compareActive])

  const trend = summary ? TREND_STYLE[summary.label] : null
  const deltaText = summary
    ? `${summary.deltaPerThousandGames >= 0 ? '+' : ''}${formatCompact(summary.deltaPerThousandGames)}/1k`
    : 'warming up'

  const otherModels = models.filter((m) => m.id !== activeDoc?.id)
  const isConverged = summary?.label === 'converged'

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1 rounded-lg border border-white/10 p-0.5 text-[11px]">
          {(['focus', 'full'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setYMode(m)}
              className={`rounded px-2 py-0.5 font-medium transition-colors ${
                yMode === m ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
              title={
                m === 'focus'
                  ? 'Scale the y-axis to the average ± spread, so real gains are visible'
                  : 'Scale to the full range, including lucky best-game spikes'
              }
            >
              {m === 'focus' ? 'Average focus' : 'Full range'}
            </button>
          ))}
        </div>
        {trend && (
          <div
            className={`rounded-lg border px-2.5 py-1 text-[11px] font-semibold tabular-nums ${trend.chip}`}
            title={`Trend over the latest buckets: ${formatCompact(summary!.deltaPerThousandGames)} score per 1,000 games`}
          >
            {trend.text} · {deltaText}
          </div>
        )}
      </div>

      <UPlotChart options={options} data={data} />

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-zinc-500">
        <span>
          <span className="mr-1 inline-block h-0.5 w-4 bg-gold-500 align-middle" />
          avg / 100 games
        </span>
        <span>
          <span className="mr-1 inline-block h-2 w-4 bg-gold-500/10 align-middle" />
          ±1σ spread
        </span>
        <span>
          <span className="mr-1 inline-block h-0.5 w-4 bg-sky-400 align-middle" />
          trajectory
        </span>
        {ceiling && (
          <span>
            <span className="mr-1 inline-block h-2 w-4 bg-emerald-500/20 align-middle" />
            expected {ceiling.label} ceiling
          </span>
        )}
        {compareActive && compareDoc && (
          <span>
            <span className="mr-1 inline-block h-0.5 w-4 bg-purple-400 align-middle" />
            {compareDoc.name}
          </span>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        {otherModels.length > 0 && (
          <label className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            Compare vs
            <select
              value={compareId ?? ''}
              onChange={(e) => setCompareId(e.target.value || null)}
              className="rounded border border-white/10 bg-surface-2 px-1.5 py-0.5 text-[11px] text-zinc-300"
            >
              <option value="">none</option>
              {otherModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {isConverged && (
          <div className="flex items-center gap-2">
            {activeDoc?.kind === 'ntuple' && planningDepth < 2 && (
              <button
                onClick={() => bridge.setPlanningDepth(2)}
                className="rounded-lg border border-white/10 px-2.5 py-1 text-[11px] font-semibold text-zinc-300 transition-colors hover:bg-white/5"
                title="Search two spawns ahead using the learned values — stronger play with no retraining"
              >
                Try planning ×2
              </button>
            )}
            {levelUpTo && activeDoc && (
              <button
                onClick={() =>
                  setLab({
                    wizardOpen: true,
                    wizardSeed: { preset: levelUpTo, fromName: activeDoc.name, autostart: true },
                  })
                }
                className="rounded-lg border border-gold-500/30 bg-gold-500/10 px-2.5 py-1 text-[11px] font-semibold text-gold-300 transition-colors hover:bg-gold-500/20"
                title="This model is near its architecture's ceiling — spin up a bigger brain that can climb higher"
              >
                Level up this brain →
              </button>
            )}
          </div>
        )}
      </div>

      {isConverged && (
        <p className="mt-2 text-[11px] leading-relaxed text-zinc-600">
          This model looks <span className="text-amber-300/80">converged near its capacity</span> —
          it has learned about all its {ceiling?.label ?? 'current'} architecture can hold. That's a
          finish line, not a failure. To climb higher, level up to a bigger preset, or raise planning
          depth for stronger play from the same weights.
        </p>
      )}
    </div>
  )
}
function SimpleSeries({
  all,
  label,
  color,
  pick,
}: {
  all: StatBucket[]
  label: string
  color: string
  pick: (b: StatBucket) => number
}) {
  const data = useMemo<uPlot.AlignedData>(
    () => [xVals(all), all.map((b) => finiteOrNull(pick(b)))],
    [all, pick],
  )
  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(
    () => ({
      legend: { show: false },
      cursor: { points: { show: false } },
      scales: { x: { time: false } },
      axes: [AXIS_STYLE, { ...AXIS_STYLE, size: 52 }],
      series: [{}, { stroke: color, width: 2, points: { show: false } }],
    }),
    [color],
  )
  return (
    <div>
      <UPlotChart options={options} data={data} />
      <div className="mt-2 text-[11px] text-zinc-500">
        <span className="mr-1 inline-block h-0.5 w-4 align-middle" style={{ background: color }} />
        {label}
      </div>
    </div>
  )
}

function RatesChart({ all }: { all: StatBucket[] }) {
  const data = useMemo<uPlot.AlignedData>(
    () => [xVals(all), all.map((b) => finiteOrNull(b.epsAvg)), all.map((b) => finiteOrNull(b.lrAvg))],
    [all],
  )
  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(
    () => ({
      legend: { show: false },
      cursor: { points: { show: false } },
      scales: { x: { time: false } },
      axes: [AXIS_STYLE, { ...AXIS_STYLE, size: 52 }],
      series: [
        {},
        { stroke: '#38bdf8', width: 2, points: { show: false } },
        { stroke: '#a3e635', width: 2, dash: [6, 3], points: { show: false } },
      ],
    }),
    [],
  )
  return (
    <div>
      <UPlotChart options={options} data={data} />
      <div className="mt-2 flex gap-4 text-[11px] text-zinc-500">
        <span>
          <span className="mr-1 inline-block h-0.5 w-4 bg-sky-400 align-middle" />ε exploration
        </span>
        <span>
          <span className="mr-1 inline-block h-0.5 w-4 bg-lime-400 align-middle" />α learning rate
        </span>
      </div>
    </div>
  )
}

function TileHistogram({ all }: { all: StatBucket[] }) {
  const { counts, total } = useMemo(() => {
    const counts = new Array(16).fill(0)
    let total = 0
    for (const b of all) {
      for (let e = 0; e < 16; e++) counts[e] += b.maxTileHist[e] ?? 0
      total += b.n
    }
    return { counts, total }
  }, [all])
  // Group everything below 128 into one bar; then 128 ... 32768.
  const low = counts.slice(0, 7).reduce((a: number, b: number) => a + b, 0)
  const bars = [
    { label: '<128', exp: 5, count: low },
    ...Array.from({ length: 9 }, (_, i) => ({
      label: formatCompact(1 << (i + 7)),
      exp: i + 7,
      count: counts[i + 7],
    })),
  ]
  const max = Math.max(1, ...bars.map((b) => b.count))
  return (
    <div>
      <div className="flex h-44 items-end gap-1.5">
        {bars.map((b) => {
          const theme = tileTheme(b.exp)
          const h = b.count === 0 ? 0 : 8 + (92 * b.count) / max
          return (
            <div key={b.label} className="flex flex-1 flex-col items-center gap-1" title={`${b.count.toLocaleString()} games ended with max tile ${b.label}`}>
              <span className="font-mono text-[10px] tabular-nums text-zinc-500">
                {b.count > 0 ? formatCompact(b.count) : ''}
              </span>
              <div
                className="w-full rounded-t transition-[height] duration-300"
                style={{ height: `${h}%`, background: b.count === 0 ? 'rgba(255,255,255,0.04)' : theme.bg, minHeight: 2 }}
              />
              <span className="text-[10px] text-zinc-500">{b.label}</span>
            </div>
          )
        })}
      </div>
      <div className="mt-2 text-[11px] text-zinc-600">
        Best tile reached per game — {total.toLocaleString()} games
      </div>
    </div>
  )
}

