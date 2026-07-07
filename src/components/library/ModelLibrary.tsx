import { useRef } from 'react'
import { useLabStore, setLab } from '../../state/labStore'
import { bridge } from '../../state/workerBridge'
import type { ModelDoc } from '../../shared/types'
import { formatAgo, formatCompact } from '../../lib/format'
import { tileTheme } from '../../lib/tileTheme'
import { getBuckets } from '../../persistence/statStore'
import { bucketMean } from '../../stats/buckets'
import { useEffect, useState } from 'react'

export function ModelLibrary() {
  const models = useLabStore((s) => s.models)
  const activeDoc = useLabStore((s) => s.activeDoc)
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between p-4 pb-3">
        <h2 className="text-sm font-semibold text-zinc-200">Models</h2>
        <div className="flex gap-1.5">
          <button
            onClick={() => fileRef.current?.click()}
            className="rounded-md px-2 py-1 text-xs font-medium text-zinc-400 transition-colors hover:bg-white/5 hover:text-zinc-200"
            title="Import a .2048model file"
          >
            Import
          </button>
          <button
            onClick={() => setLab({ wizardOpen: true })}
            className="rounded-md bg-gold-500 px-2.5 py-1 text-xs font-semibold text-zinc-950 transition-colors hover:bg-gold-400"
          >
            + New model
          </button>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".2048model"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void bridge.importFile(f)
            e.target.value = ''
          }}
        />
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto px-3 pb-4">
        {models.length === 0 && (
          <div className="mt-10 px-4 text-center">
            <div className="text-sm font-medium text-zinc-400">No models yet</div>
            <div className="mt-1 text-xs leading-relaxed text-zinc-600">
              Create one and watch it learn 2048 from scratch — it will flail at
              random first. That's the point.
            </div>
          </div>
        )}
        {models.map((m) => (
          <ModelCard key={m.id} doc={m} active={m.id === activeDoc?.id} />
        ))}
      </div>
    </div>
  )
}

function ModelCard({ doc, active }: { doc: ModelDoc; active: boolean }) {
  const running = useLabStore((s) => s.running)
  return (
    <div
      onClick={() => !active && bridge.loadModel(doc.id)}
      className={`group cursor-pointer rounded-xl border p-3 transition-colors ${
        active
          ? 'border-gold-500/40 bg-gold-500/[0.06]'
          : 'border-white/5 bg-surface-1 hover:border-white/10 hover:bg-surface-2'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-zinc-200">
            {doc.name}
            {active && running && (
              <span className="ml-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400 align-middle" />
            )}
          </div>
          <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-zinc-500">
            <span
              className={`rounded px-1 py-px font-medium ${
                doc.kind === 'ntuple'
                  ? 'bg-gold-500/10 text-gold-300'
                  : 'bg-sky-500/10 text-sky-300'
              }`}
            >
              {doc.kind === 'ntuple' ? 'N-Tuple' : 'DQN'}
            </span>
            <span>{formatCompact(doc.games)} games</span>
            {doc.forkedFrom && <span title={`Forked from ${doc.forkedFrom}`}>⑂</span>}
          </div>
        </div>
        {doc.bestExp > 0 && (
          <div
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-[10px] font-bold"
            style={{
              background: tileTheme(doc.bestExp).bg,
              color: tileTheme(doc.bestExp).fg,
            }}
            title={`Best tile: ${1 << doc.bestExp}`}
          >
            {formatCompact(1 << doc.bestExp)}
          </div>
        )}
      </div>
      <Sparkline modelId={doc.id} games={doc.games} />
      <div className="mt-2 flex items-center justify-between">
        <span className="text-[11px] text-zinc-600">updated {formatAgo(doc.updatedAt)}</span>
        <div className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
          <CardAction
            label="Fork"
            title="Copy this model's brain into a new model (stats reset)"
            onClick={(e) => {
              e.stopPropagation()
              const name = prompt('Name for the fork:', `${doc.name} fork`)
              if (name) void bridge.forkModel(doc.id, name)
            }}
          />
          <CardAction
            label="Rename"
            onClick={(e) => {
              e.stopPropagation()
              const name = prompt('New name:', doc.name)
              if (name && name !== doc.name) void bridge.renameModel(doc.id, name)
            }}
          />
          <CardAction
            label="Delete"
            danger
            onClick={(e) => {
              e.stopPropagation()
              if (confirm(`Delete "${doc.name}" and all its training history?`)) {
                void bridge.deleteModel(doc.id)
              }
            }}
          />
        </div>
      </div>
    </div>
  )
}

function CardAction({
  label,
  title,
  danger,
  onClick,
}: {
  label: string
  title?: string
  danger?: boolean
  onClick: (e: React.MouseEvent) => void
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`rounded px-1.5 py-0.5 text-[11px] font-medium transition-colors ${
        danger
          ? 'text-red-400/70 hover:bg-red-500/10 hover:text-red-300'
          : 'text-zinc-500 hover:bg-white/5 hover:text-zinc-300'
      }`}
    >
      {label}
    </button>
  )
}

function Sparkline({ modelId, games }: { modelId: string; games: number }) {
  const [points, setPoints] = useState<number[]>([])
  useEffect(() => {
    let alive = true
    void getBuckets(modelId).then((buckets) => {
      if (!alive) return
      setPoints(buckets.slice(-50).map(bucketMean))
    })
    return () => {
      alive = false
    }
  }, [modelId, games])
  if (points.length < 2) return <div className="mt-2 h-6" />
  const max = Math.max(...points, 1)
  const min = Math.min(...points, 0)
  const span = max - min || 1
  const path = points
    .map((p, i) => `${(i / (points.length - 1)) * 100},${24 - ((p - min) / span) * 22}`)
    .join(' ')
  return (
    <svg viewBox="0 0 100 24" className="mt-2 h-6 w-full" preserveAspectRatio="none">
      <polyline
        points={path}
        fill="none"
        stroke="#edc22e"
        strokeOpacity="0.7"
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}
