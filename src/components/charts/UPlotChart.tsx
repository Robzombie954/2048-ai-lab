import { useEffect, useMemo, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

export const AXIS_STYLE: uPlot.Axis = {
  stroke: '#71717a',
  grid: { stroke: 'rgba(255,255,255,0.05)', width: 1 },
  ticks: { stroke: 'rgba(255,255,255,0.1)', width: 1 },
  font: '10px Inter Variable, sans-serif',
}

interface Props {
  options: Omit<uPlot.Options, 'width' | 'height'>
  data: uPlot.AlignedData
  height?: number
}

function sanitizeData(data: uPlot.AlignedData): uPlot.AlignedData {
  const xRaw = data[0] ?? []
  const seriesCount = Math.max(1, data.length)
  const safe: (number | null)[][] = Array.from({ length: seriesCount }, () => [])
  let lastX = -Infinity

  for (let i = 0; i < xRaw.length; i++) {
    const x = xRaw[i]
    if (typeof x !== 'number' || !Number.isFinite(x) || x <= lastX) continue
    lastX = x
    safe[0].push(x)
    for (let s = 1; s < seriesCount; s++) {
      const value = data[s]?.[i]
      safe[s].push(typeof value === 'number' && Number.isFinite(value) ? value : null)
    }
  }

  if (safe[0].length === 0) {
    safe[0].push(0)
    for (let s = 1; s < seriesCount; s++) safe[s].push(null)
  }

  return safe as uPlot.AlignedData
}

export function UPlotChart({ options, data, height = 210 }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const plotRef = useRef<uPlot | null>(null)
  const optionsRef = useRef(options)
  const safeData = useMemo(() => sanitizeData(data), [data])
  const dataRef = useRef(safeData)
  optionsRef.current = options
  dataRef.current = safeData

  // Recreate the plot when height or the options object identity changes.
  // Series count and y-scale range live in options, so the y-axis toggle and
  // the compare overlay both need a rebuild — parents memoize options so this
  // only fires on real changes.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const makePlot = () =>
      new uPlot(
        { ...optionsRef.current, width: Math.max(300, el.clientWidth || 300), height },
        dataRef.current,
        el,
      )
    plotRef.current = makePlot()
    const ro = new ResizeObserver(() => {
      plotRef.current?.setSize({ width: Math.max(300, el.clientWidth || 300), height })
    })
    ro.observe(el)
    return () => {
      ro.disconnect()
      plotRef.current?.destroy()
      plotRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, options])

  useEffect(() => {
    const el = ref.current
    const plot = plotRef.current
    if (!plot || !el) return
    try {
      plot.setData(safeData)
    } catch (err) {
      console.warn('uPlot recovered from invalid chart data', err)
      plot.destroy()
      plotRef.current = new uPlot(
        { ...optionsRef.current, width: Math.max(300, el.clientWidth || 300), height },
        safeData,
        el,
      )
    }
  }, [safeData, height])

  return <div ref={ref} className="w-full" />
}
