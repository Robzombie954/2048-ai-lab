import { useEffect, useMemo, useState } from 'react'
import type { NTuplePresetId } from '../shared/types'
import { NTUPLE_PRESETS } from '../agents/ntuple/patterns'
import { runNTupleGpuBenchmark, type NTupleGpuBenchmarkResult } from '../agents/ntuple/gpuBenchmark'
import { useLabStore } from '../state/labStore'
import { formatCompact } from '../lib/format'

const PRESET_IDS = Object.keys(NTUPLE_PRESETS) as NTuplePresetId[]

function formatMs(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '—'
  if (ms < 10) return `${ms.toFixed(2)}ms`
  if (ms < 100) return `${ms.toFixed(1)}ms`
  return `${Math.round(ms).toLocaleString('en-US')}ms`
}

function formatSpeed(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '—'
  return `${formatCompact(n)}/sec`
}

function verdict(result: NTupleGpuBenchmarkResult): string {
  if (result.status !== 'ok') return result.detail
  if (result.speedup >= 1.15) return 'GPU won this tuple lookup test.'
  if (result.speedup <= 0.87) return 'CPU won this tuple lookup test.'
  return 'CPU and GPU are close on this tuple lookup test.'
}

export function GpuBenchmarkPanel() {
  const activeDoc = useLabStore((s) => s.activeDoc)
  const trainingRunning = useLabStore((s) => s.running)
  const activePreset = activeDoc?.config.kind === 'ntuple' ? activeDoc.config.preset : 'starter'
  const [preset, setPreset] = useState<NTuplePresetId>(activePreset)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<NTupleGpuBenchmarkResult | null>(null)

  useEffect(() => {
    setPreset(activePreset)
  }, [activePreset])

  const selected = useMemo(() => NTUPLE_PRESETS[preset], [preset])
  if (!activeDoc || activeDoc.config.kind !== 'ntuple') return null

  async function run() {
    setRunning(true)
    setResult(null)
    await new Promise((resolve) => setTimeout(resolve, 40))
    try {
      setResult(await runNTupleGpuBenchmark(preset))
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="w-full max-w-md rounded-xl border border-sky-400/15 bg-sky-950/[0.12] p-4 shadow-[0_18px_80px_rgba(14,165,233,0.08)]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-sky-300/80">Tuple GPU test</div>
          <div className="mt-1 text-sm font-semibold text-zinc-100">CPU vs WebGPU value lookups</div>
          <div className="mt-1 text-[11px] leading-relaxed text-zinc-500">{selected.name}</div>
        </div>
        <button
          onClick={run}
          disabled={running || trainingRunning}
          className="rounded-lg bg-sky-400 px-3 py-2 text-xs font-bold text-sky-950 transition-colors hover:bg-sky-300 disabled:cursor-wait disabled:bg-sky-900 disabled:text-sky-400"
        >
          {trainingRunning ? 'Pause first' : running ? 'Running...' : 'Benchmark'}
        </button>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2">
        {PRESET_IDS.map((id) => (
          <button
            key={id}
            onClick={() => setPreset(id)}
            disabled={running || trainingRunning}
            className={`rounded-lg border px-2 py-1.5 text-xs font-semibold transition-colors ${
              id === preset
                ? 'border-sky-300/60 bg-sky-300/15 text-sky-100'
                : 'border-white/5 bg-white/[0.03] text-zinc-500 hover:border-white/10 hover:text-zinc-300'
            }`}
          >
            {id}
          </button>
        ))}
      </div>

      {result && (
        <div className="mt-4 space-y-3">
          <div
            className={`rounded-lg border px-3 py-2 text-sm ${
              result.status === 'ok'
                ? result.speedup >= 1
                  ? 'border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-100'
                  : 'border-amber-400/20 bg-amber-400/[0.08] text-amber-100'
                : 'border-red-400/20 bg-red-400/[0.08] text-red-100'
            }`}
          >
            {verdict(result)}
          </div>

          {result.status === 'ok' && (
            <>
              <div className="grid grid-cols-3 gap-2 text-center">
                <Metric label="CPU" value={formatSpeed(result.cpuBoardsPerSec)} />
                <Metric label="GPU" value={formatSpeed(result.gpuBoardsPerSec)} />
                <Metric label="Speedup" value={`${result.speedup.toFixed(2)}x`} />
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <Metric label="GPU total" value={formatMs(result.gpuMs)} />
                <Metric label="Compute" value={formatMs(result.gpuComputeMs)} />
                <Metric label="Readback" value={formatMs(result.gpuReadbackMs)} />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-zinc-500">
                <span>{formatCompact(result.boards)} boards</span>
                <span>checksum diff {result.checksumDiff.toExponential(2)}</span>
              </div>
              <div className="truncate text-[11px] text-zinc-600" title={result.detail}>
                {result.detail}
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/5 bg-black/20 px-2 py-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-600">{label}</div>
      <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-zinc-100">{value}</div>
    </div>
  )
}



