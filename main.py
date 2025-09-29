import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from pymatgen.core.composition import Composition
from fastapi.middleware.cors import CORSMiddleware

# Load trained model
model = joblib.load("energy_model_91.pkl")

app = FastAPI(title="Material Discovery API")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later to ["http://localhost:3000", "https://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema (all fields required from user)
class Features(BaseModel):
    energy_per_atom: float
    density_atomic: float
    efermi: float
    volume: float
    density: float
    band_gap: float
    chemsys: str
    is_magnetic: bool
    ordering: str
    types_of_magnetic_species: str
    is_metal: bool
    total_magnetization: float
    total_magnetization_normalized_vol: float
    total_magnetization_normalized_formula_units: float
    num_magnetic_sites: int
    num_unique_magnetic_sites: int
    formation_energy_per_atom: float
    elements: Dict[str, int]   # Example: {"Ag": 1, "Te": 1}


@app.post("/predict")
def predict_energy(data: Features):
    features_dict = data.dict()
    element_ratio_dict = features_dict.pop("elements")

    # Derived features (still computed by backend)
    features_dict["gap_density_ratio"] = features_dict["band_gap"] / (features_dict["density"] + 1e-5)
    features_dict["num_elements"] = len(element_ratio_dict)
    features_dict["band_gap_x_density"] = features_dict["band_gap"] * features_dict["density"]
    features_dict["efermi_x_volume"] = features_dict["efermi"] * features_dict["volume"]

    # DataFrame for model
    df_input = pd.DataFrame([features_dict])

    # Prediction
    predicted_energy = model.predict(df_input)[0]

    # Material formula (from elements)
    formula = Composition(element_ratio_dict).reduced_formula

    # Stability assessment
    if predicted_energy <= 0.1:
        stability = "✅ Stable (Thermodynamic Ground State)"
    elif predicted_energy <= 0.05:
        stability = "🟢 Likely Stable (within thermal noise)"
    elif predicted_energy <= 0.1:
        stability = "⚠️ Metastable (can be synthesized)"
    else:
        stability = "❌ Not Stable (unlikely to exist)"

    return {
        "material": formula,
        "input_features": features_dict,
        "predicted_energy_above_hull": round(float(predicted_energy), 4),
        "stability": stability
    }
