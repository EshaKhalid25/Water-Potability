import { useState } from 'react'

function App() {
  const [formData, setFormData] = useState({
    ph: 7.2,
    Hardness: 180,
    Solids: 15000,
    Chloramines: 7.1,
    Sulfate: 330,
    Conductivity: 420,
    Organic_carbon: 12,
    Trihalomethanes: 65,
    Turbidity: 3.5
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error("Error connecting to API:", error)
    }
    setLoading(false)
  }

  return (
    <div className="container">
      <h1>💧 Water Potability Predictor</h1>
      
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {Object.keys(formData).map((key) => (
            <div className="input-group" key={key}>
              <label>{key}</label>
              <input 
                type="number" 
                step="any" 
                name={key} 
                value={formData[key]} 
                onChange={handleChange} 
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
          <p>Model Confidence: <strong>{result.confidence * 100}%</strong></p>
        </div>
      )}
    </div>
  )
}

export default App
