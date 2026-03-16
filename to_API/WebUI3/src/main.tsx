import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import 'katex/dist/katex.min.css' // Latex styles
// Highlight.js styles will be handled by a specific theme or custom CSS
import 'highlight.js/styles/atom-one-dark.css' 

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
