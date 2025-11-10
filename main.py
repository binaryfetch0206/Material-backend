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

HBAR = 1.054571817e-34  # J·s
ME = 9.10938356e-31     # kg
QE = 1.602176634e-19    # J/eV
NA = 6.02214076e23
ANG3_TO_M3 = 1e-30      # 1 Å³ = 1e-30 m³

# ===================================================
# 🧩 Elemental Reference Data
# ===================================================
ELEMENT_DATA = {
    "H":  {"atomic_mass": 1.008, "pauling_chi": 2.20, "valence_e": 1,  "cohesive_e": 0.5,  "unpaired_e": 1},
    "O":  {"atomic_mass": 15.999,"pauling_chi": 3.44, "valence_e": 6,  "cohesive_e": 2.6,  "unpaired_e": 0},
    "Fe": {"atomic_mass": 55.845,"pauling_chi": 1.83, "valence_e": 8,  "cohesive_e": 4.28, "unpaired_e": 4},
    "Co": {"atomic_mass": 58.933,"pauling_chi": 1.88, "valence_e": 9,  "cohesive_e": 4.39, "unpaired_e": 3},
    "Ni": {"atomic_mass": 58.693,"pauling_chi": 1.91, "valence_e": 10, "cohesive_e": 4.44, "unpaired_e": 2},
    "Cu": {"atomic_mass": 63.546,"pauling_chi": 1.90, "valence_e": 11, "cohesive_e": 3.49, "unpaired_e": 1},
    "Ag": {"atomic_mass": 107.8682,"pauling_chi": 1.93,"valence_e": 1, "cohesive_e": 2.95, "unpaired_e": 0},
    "Au": {"atomic_mass": 196.967,"pauling_chi": 2.54,"valence_e": 1,  "cohesive_e": 3.81, "unpaired_e": 1},
    "Te": {"atomic_mass": 127.60,"pauling_chi": 2.10, "valence_e": 6,  "cohesive_e": 1.9,  "unpaired_e": 0},
    "Si": {"atomic_mass": 28.085,"pauling_chi": 1.90, "valence_e": 4,  "cohesive_e": 4.63, "unpaired_e": 0},
    "Al": {"atomic_mass": 26.982,"pauling_chi": 1.61, "valence_e": 3,  "cohesive_e": 3.39, "unpaired_e": 0},
    "C":  {"atomic_mass": 12.011,"pauling_chi": 2.55, "valence_e": 4,  "cohesive_e": 7.37, "unpaired_e": 0},
    "N":  {"atomic_mass": 14.007,"pauling_chi": 3.04, "valence_e": 5,  "cohesive_e": 2.3,  "unpaired_e": 1},
    "Zn": {"atomic_mass": 65.38, "pauling_chi": 1.65, "valence_e": 2,  "cohesive_e": 1.35, "unpaired_e": 0},
    "Mg": {"atomic_mass": 24.305,"pauling_chi": 1.31, "valence_e": 2,  "cohesive_e": 1.51, "unpaired_e": 0},
    "Na": {"atomic_mass": 22.989,"pauling_chi": 0.93, "valence_e": 1,  "cohesive_e": 1.13, "unpaired_e": 1},
    "K":  {"atomic_mass": 39.098,"pauling_chi": 0.82, "valence_e": 1,  "cohesive_e": 0.93, "unpaired_e": 1},
    "Ca": {"atomic_mass": 40.078,"pauling_chi": 1.00, "valence_e": 2,  "cohesive_e": 1.84, "unpaired_e": 0},
    "Ti": {"atomic_mass": 47.867,"pauling_chi": 1.54, "valence_e": 4,  "cohesive_e": 4.85, "unpaired_e": 2},
}

# ===================================================
# 🔬 Helper Functions for Hybrid Predictor
# ===================================================
def parse_chem_system(chemsys):
    import re
    chemsys = chemsys.replace("-", " ").replace("_", " ")
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", chemsys)
    elems = [(el, int(num) if num else 1) for el, num in parts if el]
    return elems or [("Ag",1),("Te",1)]

def weighted_avg(elems, key):
    total, count = 0, 0
    for el, n in elems:
        data = ELEMENT_DATA.get(el, {"atomic_mass":50,"pauling_chi":1.5,"valence_e":1,"cohesive_e":2,"unpaired_e":0})
        total += data[key]*n
        count += n
    return total/count

def estimate_density(elems, vol_A3):
    vol_m3 = vol_A3 * ANG3_TO_M3
    mass_g = sum(ELEMENT_DATA.get(el,{"atomic_mass":50})["atomic_mass"]*n for el,n in elems)/NA
    vol_cm3 = vol_m3 * 1e6
    return mass_g/vol_cm3

def band_gap_formula(elems, material_type, inputs):
    chi_vals = [ELEMENT_DATA.get(e,{"pauling_chi":1.7})["pauling_chi"] for e,_ in elems]
    delta_chi = abs(max(chi_vals)-min(chi_vals))
    Eg = 1.9*(delta_chi**2)
    mt = (material_type or "").lower()
    if "metal" in mt or "semimetal" in mt: Eg *= 0.2
    if "insulator" in mt: Eg *= 1.6
    if inputs.get("Transparency","").lower()=="high": Eg *= 1.25
    if inputs.get("Conductivity","").lower()=="high": Eg *= 0.45
    return round(Eg,4)

def fermi_energy(elems, vol_A3):
    vol_m3 = vol_A3 * ANG3_TO_M3
    valence = sum(ELEMENT_DATA.get(el,{"valence_e":1})["valence_e"]*n for el,n in elems)
    n_e = valence/vol_m3
    EF = ((HBAR**2)/(2*ME))*(3*math.pi**2*n_e)**(2/3)/QE
    return round(EF,3)

def formation_energy(elems, inputs):
    mean_coh = weighted_avg(elems,"cohesive_e")
    chi_vals = [ELEMENT_DATA.get(e,{"pauling_chi":1.7})["pauling_chi"] for e,_ in elems]
    chi_max, chi_min = max(chi_vals), min(chi_vals)
    ionic_char = abs(chi_max-chi_min)/(chi_max+chi_min+1e-9)
    stability = inputs.get("Formation Stability","").lower()
    bf = 0.75 if "high" in stability else 0.6 if "moderate" in stability else 0.45
    FE = -bf*mean_coh*(1-0.5*ionic_char)
    return round(FE,4)

def energy_per_atom(elems, FE):
    mean_coh = weighted_avg(elems,"cohesive_e")
    return round(FE - 0.2*mean_coh,4)

def magnetization(elems, inputs, vol_A3):
    total_up = sum(ELEMENT_DATA.get(el,{"unpaired_e":0})["unpaired_e"]*n for el,n in elems)
    mag_type = inputs.get("Magnetic Type","").lower()
    order = 0.9 if "ferro" in mag_type else 0.6 if "ferri" in mag_type else 0.05
    mu = total_up*order
    mu_norm = round(mu/(vol_A3 if vol_A3>0 else 1),6)
    return round(mu,3), mu_norm

def hybrid_property_predictor_v2(user_input):
    elems = parse_chem_system(user_input.get("Chemical System","Ag2Te"))
    vol_A3 = float(user_input.get("Volume",74.76))
    density_user = user_input.get("Density",None)
    density = float(density_user) if isinstance(density_user,(int,float)) else estimate_density(elems,vol_A3)
    
    Eg = band_gap_formula(elems, user_input.get("Material Type","Semimetal"), user_input)
    EF = fermi_energy(elems, vol_A3)
    FE = formation_energy(elems, user_input)
    EA = energy_per_atom(elems, FE)
    
    density_atomic = round(density*0.96,4)
    mu, mu_norm = magnetization(elems, user_input, vol_A3)
    med = round(mu*density_atomic/10,4)
    gdr = round(Eg/(density+1e-9),4)
    is_metal = 1 if Eg<0.3 else 0
    is_mag = 1 if mu>0.1 else 0
    ordering = "FM" if is_mag else "NM"
    
    return {
        "chemsys":user_input.get("Chemical System","Ag2Te"),
        "elements_input":[e for e,_ in elems],
        "density":round(density,4),
        "density_atomic":density_atomic,
        "band_gap":Eg,
        "efermi_eV":EF,
        "formation_energy_per_atom":FE,
        "energy_per_atom":EA,
        "volume_A3_per_cell":vol_A3,
        "is_metal":is_metal,
        "is_magnetic":is_mag,
        "ordering":ordering,
        "total_magnetization_muB_per_fu":mu,
        "total_magnetization_normalized_vol":mu_norm,
        "magnetic_energy_density":med,
        "gap_density_ratio":gdr,
    }

# ===================================================
# 🧩 Route: Quick Predict using Physics (No ML)
# ===================================================
@app.post("/materials/quick_predict")
def quick_predict(user_input: dict):
    """
    Quick hybrid physics-based material property estimation.
    Accepts Chemical System, Density, Volume, Material Type, etc.
    """
    try:
        result = hybrid_property_predictor_v2(user_input)
        return {"success": True, "physics_based_properties": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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
# ===================================================
# 4️⃣ Physics + Scientific Estimation Functions (for /materials/auto_features)
# ===================================================
def estimate_density(user_input):
    mat_type = user_input.get("Material Type", "Metal").lower()
    wt_type = user_input.get("Weight Type", "Medium").lower()
    base_density = {"metal": 7.8, "ceramic": 3.5, "polymer": 1.2, "semiconductor": 5.0}.get(mat_type, 5.0)
    weight_adj = {"light": 0.7, "medium": 1.0, "heavy": 1.4}.get(wt_type, 1.0)
    return np.round(np.clip(base_density * weight_adj, 0.5, 15), 3)


def estimate_volume(density, molar_mass=60):
    return np.round(np.clip(molar_mass / density * 2, 5, 200), 3)


def estimate_band_gap(user_input):
    mat_type = user_input.get("Material Type", "Metal").lower()
    if mat_type == "metal":
        return 0.0
    if mat_type == "semiconductor":
        semi_type = user_input.get("Semiconductor Type", "n-type").lower()
        return 1.1 if semi_type == "n-type" else 1.8
    if mat_type == "insulator":
        return 4.5
    return 2.0


def estimate_efermi(band_gap, is_metal):
    return np.round(np.random.uniform(-1, 1), 3) if is_metal else np.round(band_gap / 2 * np.random.choice([-1, 1]), 3)


def estimate_magnetism(user_input):
    mag_type = user_input.get("Magnetic Type", "").lower()
    mag_strength = user_input.get("Magnetization Strength", "Medium").lower()
    is_mag = 1 if mag_type in ["ferromagnetic", "ferrimagnetic"] else 0
    ordering = "FM" if is_mag else "NM"
    base_mag = {"low": 0.3, "medium": 1.5, "high": 3.0}.get(mag_strength, 1.0)
    total_mag = base_mag if is_mag else 0.0
    return is_mag, ordering, total_mag


def scientific_estimates(user_input):
    density = estimate_density(user_input)
    volume = estimate_volume(density)
    band_gap = estimate_band_gap(user_input)
    is_metal = 1 if band_gap < 0.3 else 0
    efermi = estimate_efermi(band_gap, is_metal)
    is_magnetic, ordering, total_mag = estimate_magnetism(user_input)
    return {
        "density": density, "volume": volume, "band_gap": band_gap,
        "efermi": efermi, "is_metal": is_metal,
        "is_magnetic": is_magnetic, "ordering": ordering, "total_magnetization": total_mag,
        "density_atomic": 4.75, "num_magnetic_sites": 3.6, "num_unique_magnetic_sites": 3.0,
        "types_of_magnetic_species": "['Fe']",
        "total_magnetization_normalized_vol": 0.86,
        "total_magnetization_normalized_formula_units": 0.38,
        "energy_per_atom": 4.34,
        "energy_above_hull": 3.04,
        "formation_energy_per_atom": 2.83
    }


# ===================================================
# 5️⃣ Route 2: Auto-generate scientific + ML hybrid properties
# ===================================================
@app.post("/materials/auto_features")
def auto_features(user_input: dict):
    """
    Accepts partial material information from frontend like:
    {
      "Material Type": "Semiconductor",
      "Weight Type": "Medium",
      "Magnetic Type": "Diamagnetic",
      "Magnetization Strength": "Low"
    }
    """
    try:
        results = scientific_estimates(user_input)
        return {
            "success": True,
            "auto_generated_features": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
