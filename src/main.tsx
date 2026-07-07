import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import App from './App'
import { bridge } from './state/workerBridge'
import { useLabStore } from './state/labStore'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

void bridge.boot()

if (import.meta.env.DEV) {
  // Store + bridge inspection hooks for debugging sessions.
  ;(window as unknown as Record<string, unknown>).__lab = useLabStore
  ;(window as unknown as Record<string, unknown>).__bridge = bridge
}
