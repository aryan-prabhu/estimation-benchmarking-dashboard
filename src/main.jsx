import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import StateEstimationDashboard from './StateEstimationDashboard.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <StateEstimationDashboard />
  </StrictMode>,
)
