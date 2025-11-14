import { useState } from "react";

function App() {
  const [inputs, setInputs] = useState({
    Crop_Year: "",
    Area: "",
    Production: "",
    Annual_Rainfall: "",
    Fertilizer: "",
    Pesticide: "",
    HUMPIDITY: "",
    AVG_TEMPERATURE: ""
  });
  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        Crop_Year: parseInt(inputs.Crop_Year),
        Area: parseFloat(inputs.Area),
        Production: parseFloat(inputs.Production),
        Annual_Rainfall: parseFloat(inputs.Annual_Rainfall),
        Fertilizer: parseFloat(inputs.Fertilizer),
        Pesticide: parseFloat(inputs.Pesticide),
        HUMPIDITY: parseFloat(inputs.HUMPIDITY),
        AVG_TEMPERATURE: parseFloat(inputs.AVG_TEMPERATURE)
      }),
    });
    const data = await res.json();
    setPrediction(data.predicted_yield);
  };

  return (
    <div className="p-6 max-w-lg mx-auto">
      <h1 className="text-2xl font-bold mb-4">Crop Yield Predictor 🌱</h1>
      <form onSubmit={handleSubmit} className="space-y-3">
        {Object.keys(inputs).map((key) => (
          <input
            key={key}
            name={key}
            placeholder={key}
            value={inputs[key]}
            onChange={handleChange}
            className="border p-2 w-full rounded"
          />
        ))}
        <button type="submit" className="bg-green-600 text-white px-4 py-2 rounded">
          Predict
        </button>
      </form>
      {prediction !== null && (
        <div className="mt-6 text-xl font-bold">
          Predicted Yield: {prediction.toFixed(2)} kg/hectare
        </div>
      )}
    </div>
  );
}

export default App;
