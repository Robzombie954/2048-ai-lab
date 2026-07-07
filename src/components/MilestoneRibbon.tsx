import { useLabStore } from '../state/labStore'
import { formatCompact, formatDuration } from '../lib/format'
import { tileTheme } from '../lib/tileTheme'

const RIBBON_EXPS = [7, 8, 9, 10, 11, 12, 13, 14, 15] // 128 … 32768

export function MilestoneRibbon() {
  const milestones = useLabStore((s) => s.milestones)
  const activeDoc = useLabStore((s) => s.activeDoc)
  if (!activeDoc) return null
  const byExp = new Map(milestones.map((m) => [m.exp, m]))

  return (
    <div className="border-t border-white/5 bg-surface-1/60 px-4 py-2.5">
      <div className="mx-auto flex max-w-5xl items-center gap-2 overflow-x-auto">
        <span className="mr-1 shrink-0 text-[11px] font-medium uppercase tracking-wide text-zinc-600">
          Firsts
        </span>
        {RIBBON_EXPS.map((exp) => {
          const m = byExp.get(exp)
          const theme = tileTheme(exp)
          return (
            <div
              key={exp}
              className={`flex shrink-0 items-center gap-1.5 rounded-lg border px-2 py-1 ${
                m ? 'border-transparent' : 'border-dashed border-white/10 opacity-40'
              }`}
              style={m ? { background: 'rgba(255,255,255,0.03)' } : undefined}
              title={
                m
                  ? `First ${1 << exp}: game ${m.game + 1}, after ${formatDuration(m.wallMs)} of training`
                  : `${1 << exp} — not reached yet`
              }
            >
              <span
                className="flex h-6 w-8 items-center justify-center rounded text-[10px] font-bold"
                style={{
                  background: m ? theme.bg : 'rgba(255,255,255,0.06)',
                  color: m ? theme.fg : '#52525b',
                  boxShadow: m && exp >= 11 ? theme.boxShadow : undefined,
                }}
              >
                {formatCompact(1 << exp)}
              </span>
              {m && (
                <span className="font-mono text-[10px] tabular-nums text-zinc-500">
                  g{formatCompact(m.game + 1)}
                </span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
