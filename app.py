# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
import numpy as np

# --- CONFIG: paths (assume models/ folder is in repo root) ---
MODEL_PATH = os.path.join("models", "xgboost_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
FEATURES_PATH = os.path.join("models", "features.txt")

# --- Load model, scaler, features ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Features list not found at {FEATURES_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, "r") as f:
    internal_features = [line.strip() for line in f.readlines() if line.strip()]

# --- Human-readable names mapping (change names if you prefer) ---
# Map readable_name -> internal feature name (internal_features must remain in same order)
readable_to_internal = {
    "Altitude_factor": "op_setting1",
    "Throttle_resolver_angle": "op_setting2",
    "Mach_temperature_index": "op_setting3",
    "Fan_inlet_temperature_T2": "s1",
    "LPC_outlet_temperature_T24": "s2",
    "HPC_outlet_temperature_T30": "s3",
    "LPT_outlet_temperature_T50": "s4",
    "Fan_inlet_pressure_P2": "s5",
    "LPC_outlet_pressure_P15": "s6",
    "HPC_outlet_pressure_P30": "s7",
    "Physical_fan_speed_Nf": "s8",
    "Physical_core_speed_Nc": "s9",
    "Engine_pressure_ratio_EPR": "s10",
    "Bypass_duct_exit_pressure_Ps30": "s11",
    "Fuel_flow_ratio_to_Ps30": "s12",
    "Corrected_fan_speed_Wf": "s13",
    "Corrected_core_speed": "s14",
    "Bypass_ratio_BPR": "s15",
    "Burner_fuel_air_ratio_FARB": "s16",
    "Bleed_enthalpy": "s17",
    "Demanded_fan_speed": "s18",
    "Demanded_corrected_fan_speed": "s19",
    "HPT_coolant_bleed": "s20",
    "LPT_coolant_bleed": "s21",
}

# build reverse map for validation convenience
internal_set = set(internal_features)
readable_set = set(readable_to_internal.keys())

app = FastAPI(title="PdM RUL API (XGBoost)")


class InputData(BaseModel):
    data: dict  # accept dict of feature:value. keys can be internal OR readable names


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/predict")
def predict(payload: InputData):
    user_dict = payload.data
    if not isinstance(user_dict, dict):
        raise HTTPException(
            status_code=400,
            detail="`data` must be an object/dict of feature:value pairs",
        )

    # Build input vector in the required order (internal_features)
    row = []
    missing = []
    for feat in internal_features:
        # priority: user may provide readable name that maps to this internal feature
        val = None
        # 1) direct internal name
        if feat in user_dict:
            val = user_dict.get(feat)
        else:
            # 2) see if user provided a readable name mapping to this feat
            # find readable keys that map to this internal feat
            # (this is O(n) over mapping but mapping is small)
            readable_matches = [r for r, i in readable_to_internal.items() if i == feat]
            found = False
            for r in readable_matches:
                if r in user_dict:
                    val = user_dict.get(r)
                    found = True
                    break
        # If still None, default to 0.0 (you can change to throw error instead)
        if val is None:
            # record missing but continue with zero fill
            missing.append(feat)
            val = 0.0
        # ensure numeric
        try:
            row.append(float(val))
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"Feature '{feat}' has non-numeric value: {val}"
            )

    # Convert to numpy array and scale
    x = np.array(row).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]

    return {"Predicted_RUL": float(pred), "missing_filled_with_zero": missing}
