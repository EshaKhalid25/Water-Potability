import { useState } from 'react'

// 1. Define strict limits based on your actual machine learning dataset
const FIELD_LIMITS = {
  ph: { min: 0, max: 14 },
  Hardness: { min: 45, max: 325 },
  Solids: { min: 320, max: 61230 },
  Chloramines: { min: 0.3, max: 13.2 },
  Sulfate: { min: 129, max: 482 },
  Conductivity: { min: 181, max: 754 },
  Organic_carbon: { min: 2, max: 29 },
  Trihalomethanes: { min: 0.7, max: 125 },
  Turbidity: { min: 1.4, max: 6.8 }
};

function App() {
  const [formData, setFormData] = useState({
    ph: '',
    Hardness: '',
    Solids: '',
    Chloramines: '',
    Sulfate: '',
    Conductivity: '',
    Organic_carbon: '',
    Trihalomethanes: '',
    Turbidity: ''
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    // Convert values to numbers for the API
    const payload = {}
    for (let key in formData) {
      payload[key] = parseFloat(formData[key])
    }

    try {
      const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error("Error connecting to API:", error)
    }
    setLoading(false)
  }

  // Helper function to format the label text for the UI
  const formatLabel = (key) => {
    if (key === 'ph') return 'pH';
    return key.replace('_', ' '); // Replaces the underscore with a space
  }

  return (
    <div className="container">
      <h1>💧 Water Potability Predictor</h1>
      
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {Object.keys(formData).map((key) => (
            <div className="input-group" key={key}>
              <label>
                {/* Use the helper function here to clean up the display text */}
                {formatLabel(key)} <span style={{ fontSize: '0.75rem', color: '#7f8c8d' }}>
                  ({FIELD_LIMITS[key].min} - {FIELD_LIMITS[key].max})
                </span>
              </label>
              <input 
                type="number" 
                step="any" 
                name={key} 
                value={formData[key]} 
                onChange={handleChange} 
                min={FIELD_LIMITS[key].min}
                max={FIELD_LIMITS[key].max}
                required 
              />
            </div>
          ))}
        </div>
        <button type="submit" className="btn" disabled={loading}>
          {loading ? "Predicting..." : "Predict Water Quality"}
        </button>
      </form>

      {result && (
        <div className="result-card">
          <h2 className={result.prediction === 1 ? 'potable' : 'not-potable'}>
            {result.prediction === 1 ? '✓ POTABLE' : '✕ NOT POTABLE'}
          </h2>
          <p>Model Confidence: <strong>{Math.round(result.confidence * 100)}%</strong></p>
        </div>
      )}
    </div>
  )
}

export default App
