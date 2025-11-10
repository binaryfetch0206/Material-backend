import numpy as np
import pandas as pd
import joblib
import os


# ===================================================
# 1️⃣ Physics-Inspired Estimation Functions
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


# ===================================================
# 2️⃣ Scientific Defaults
# ===================================================
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
        "types_of_magnetic_species": "['Fe']",  # ✅ make string not float
        "total_magnetization_normalized_vol": 0.86,
        "total_magnetization_normalized_formula_units": 0.38,
        "energy_per_atom": 4.34,
        "energy_above_hull": 3.04,
        "formation_energy_per_atom": 2.83
    }


# ===================================================
# 3️⃣ Model Hierarchy
# ===================================================
MODEL_TIERS = [
    [],
    ["density_atomic", "num_magnetic_sites", "num_unique_magnetic_sites", "types_of_magnetic_species"],
    ["total_magnetization_normalized_vol", "total_magnetization_normalized_formula_units", "energy_per_atom"],
    ["energy_above_hull", "formation_energy_per_atom"],
]

MODEL_PATHS = {
    "density_atomic": "trained_models/density_atomic_HistGBR_reg.pkl",
    "num_unique_magnetic_sites": "trained_models/num_unique_magnetic_sites_RandomForest_reg.pkl",
    "num_magnetic_sites": "trained_models/num_magnetic_sites_HistGBR_reg.pkl",
    "types_of_magnetic_species": "trained_models/types_of_magnetic_species_RandomForest_clf.pkl",
    "total_magnetization_normalized_vol": "trained_models/total_magnetization_normalized_vol_HistGBR_reg.pkl",
    "total_magnetization_normalized_formula_units": "trained_models/total_magnetization_normalized_formula_units_HistGBR_reg.pkl",
    "energy_per_atom": "trained_models/energy_per_atom_RandomForest_reg.pkl",
    "energy_above_hull": "trained_models/energy_above_hull_RandomForest_reg.pkl",
    "formation_energy_per_atom": "trained_models/formation_energy_per_atom_HistGBR_reg.pkl",
}


# ===================================================
# 4️⃣ Cascaded Model Runner with Debug
# ===================================================
def run_cascaded_models(user_input):
    sci_vals = scientific_estimates(user_input)
    all_predictions = {
        "formula_pretty": user_input.get("Formula Pretty", "Fe2O3"),
        "formula_anonymous": user_input.get("Formula Anonymous", "A2B3"),
        "chemsys": user_input.get("Chemical System", "Fe-O"),
        **sci_vals,
    }

    print("\n✅ Tier 1 initialized with scientific estimates:")
    for k, v in sci_vals.items():
        print(f"   • {k}: {v}")

    input_df = pd.DataFrame([all_predictions])
    ml_used, sci_used_models = [], []

    for tier_i, models in enumerate(MODEL_TIERS, start=1):
        print(f"\n🔹 Running Tier {tier_i} models...")
        for model_name in models:
            model_path = MODEL_PATHS.get(model_name)
            if not model_path or not os.path.exists(model_path):
                print(f"⚙️  {model_name}: model missing → scientific fallback")
                all_predictions[model_name] = sci_vals[model_name]
                input_df[model_name] = sci_vals[model_name]
                sci_used_models.append(model_name)
                continue

            try:
                model = joblib.load(model_path)
                feature_names = getattr(model, "feature_names_in_", input_df.columns)
                X = input_df.copy()

                # Track scientific feature usage
                used_tier1_features = [
                    f for f in feature_names
                    if f in sci_vals and X[f].iloc[0] == sci_vals[f]
                ]

                pred = model.predict(X[feature_names])
                val = pred[0] if isinstance(pred, (list, np.ndarray)) else pred

                # clean up type conversion
                if hasattr(model, "_estimator_type") and model._estimator_type == "classifier":
                    val_clean = str(val)
                else:
                    try:
                        val_clean = float(val)
                    except Exception:
                        val_clean = str(val)

                msg = f"✅ {model_name}: {val_clean}"
                if used_tier1_features:
                    msg += f"  (used Tier 1 scientific inputs: {used_tier1_features})"
                print(msg)

                all_predictions[model_name] = val_clean
                input_df[model_name] = val_clean
                ml_used.append(model_name)

            except Exception as e:
                print(f"⚠️  {model_name}: prediction failed ({e}) → scientific fallback")
                val = sci_vals[model_name]
                all_predictions[model_name] = val
                input_df[model_name] = val
                sci_used_models.append(model_name)

    # update magnetism flags
    all_predictions["is_magnetic"] = 1 if all_predictions.get("total_magnetization", 0) > 0.1 else 0
    all_predictions["ordering"] = "FM" if all_predictions["is_magnetic"] else "NM"

    print("\n📊 Summary:")
    print(f"   ML predicted: {len(ml_used)} → {', '.join(ml_used) if ml_used else 'None'}")
    print(f"   Scientific fallback: {len(sci_used_models)} → {', '.join(sci_used_models) if sci_used_models else 'None'}")

    return pd.DataFrame([all_predictions])


# ===================================================
# 5️⃣ Example Run
# ===================================================
if __name__ == "__main__":
    user_input_example = {
    "Elasticity": "Medium",
    "Plasticity": "Medium",
    "Ductility": "Medium",
    "Toughness": "Medium",
    "Brittleness": "Medium",
    "Hardness": "Medium",
    "Strength": "Medium",
    "Flexibility": "Medium",
    "Weight Type": "Medium",                # density 5.0 -> not very heavy
    "Thermal Conductivity": "Medium",
    "Specific Heat": "Medium",
    "Thermal Expansion": "Medium",
    "Conductivity": "Medium",               # small gap -> limited conduction
    "Resistivity": "Medium",
    "Dielectric Constant": "Medium",
    "Semiconductor Type": "n-type",         # keep n-type if prior knowledge says so
    "Electrical Breakdown Strength": "Medium",
    "Corrosion Resistance": "Medium",
    "Reactivity": "Medium",
    "pH Stability": "Medium",
    "Oxidation Potential": "Medium",
    "Magnetic Type": "Diamagnetic",
    "Magnetization Strength": "Low",
    "Magnetic Susceptibility": "Low",
    "Refractive Index": "Medium",
    "Transparency": "Low",                  # narrow gap -> mostly opaque except IR/near-IR
    "Absorption Coefficient": "Medium",
    "Color Appearance": "Grayish",
    "Chemical System": "Ag-Te",
    "Material Type": "Narrow-gap Semiconductor",
    "Density": "Medium",
    "Volume": "Low",
    "Processing Type": "Sputtered / Thin-film (possible)",
    "Structural Stability": "Low",          # positive formation energy -> unstable
    "Formation Stability": "Unstable",
    "Magnetic Ordering": "Non-Magnetic",
    "Electronic Nature": "Semiconductor",
    "Bond Strength": "Weak/Moderate"
}

    results = run_cascaded_models(user_input_example)
    print("\n🏁 Final Predicted Material Properties:")
    print(results.T)
