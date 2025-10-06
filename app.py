# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from typing import Tuple, Optional

app = FastAPI(title="PdM RUL API (XGBoost)")

# -------------------------
# Helper: locate model files
# -------------------------
def find_artifact_paths() -> Tuple[str, str, str]:
    """
    Try common locations for model artifacts.
    Return (model_path, scaler_path, features_path).
    Raises FileNotFoundError if none found.
    """
    candidates = [
        # repo root
        ("xgboost_model.pkl", "scaler.pkl", "features.txt"),
        # models/ subfolder
        (os.path.join("models", "xgboost_model.pkl"),
         os.path.join("models", "scaler.pkl"),
         os.path.join("models", "features.txt")),
    ]

    for model_p, scaler_p, feat_p in candidates:
        if os.path.exists(model_p) and os.path.exists(scaler_p) and os.path.exists(feat_p):
            return model_p, scaler_p, feat_p

    # not found
    raise FileNotFoundError(
        "Model artifacts not found. Expected either root: "
        "'xgboost_model.pkl','scaler.pkl','features.txt' or in 'models/' subfolder."
    )

# -------------------------
# Startup diagnostics
# -------------------------
print("=== STARTUP: PdM RUL API ===")
print("cwd:", os.getcwd())
print("files in cwd:", os.listdir())
# if there's a models folder, print it too
if os.path.exists("models"):
    try:
        print("files in ./models:", os.listdir("models"))
    except Exception as e:
        print("could not list ./models:", str(e))

# -------------------------
# Load artifacts (safe)
# -------------------------
try:
    MODEL_PATH, SCALER_PATH, FEATURES_PATH = find_artifact_paths()
    print(f"Using MODEL_PATH={MODEL_PATH}, SCALER_PATH={SCALER_PATH}, FEATURES_PATH={FEATURES_PATH}")
except FileNotFoundError as e:
    # re-raise as RuntimeError so container shows it in logs and app continues to start
    print("ERROR:", str(e))
    # we still create the app but will return 500 on predict until fixed
    MODEL_PATH = SCALER_PATH = FEATURES_PATH = None

model = None
scaler = None
internal_features = []

if MODEL_PATH and SCALER_PATH and FEATURES_PATH:
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, "r") as f:
            internal_features = [line.strip() for line in f.readlines() if line.strip()]
        print("Model and scaler loaded successfully. Number of features:", len(internal_features))
    except Exception as e:
        # If loading fails, print error. We'll return 500 on predict.
        print("Failed to load model/scaler/features:", str(e))
        model = None
        scaler = None
        internal_features = []

# -------------------------
# Human-readable mapping
# -------------------------
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

# Inverse map (internal -> list of readable names)
internal_to_readables = {}
for r, i in readable_to_internal.items():
    internal_to_readables.setdefault(i, []).append(r)

# -------------------------
# Request model
# -------------------------
class InputData(BaseModel):
    data: dict

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance RUL API. Use /predict to POST sensor values.",
        "note": "This API expects feature names (s1..s21 and op_setting1..3) or human-readable names.",
        "health": "/health",
        "info": "/info"
    }

@app.get("/health")
def health():
    ok = model is not None and scaler is not None and len(internal_features) > 0
    return {
        "status": "ok" if ok else "error",
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "features_count": len(internal_features),
    }

@app.get("/info")
def info():
    # Provide simple feature info (name + readable names if available)
    features_info = []
    for feat in internal_features:
        features_info.append({
            "internal_name": feat,
            "readable_aliases": internal_to_readables.get(feat, [])
        })
    return {
        "model_loaded": model is not None,
        "features": features_info,
        "note": "You can supply either internal_name or one of readable_aliases in /predict payload."
    }

@app.post("/predict")
def predict(payload: InputData):
    if model is None or scaler is None or not internal_features:
        # Return 500 with clear message so Railway logs show the reason
        raise HTTPException(status_code=500, detail="Model or scaler not loaded on server. Check logs.")

    user_dict = payload.data
    if not isinstance(user_dict, dict):
        raise HTTPException(status_code=400, detail="`data` must be an object/dict of feature:value pairs")

    row = []
    missing = []
    for feat in internal_features:
        val: Optional[float] = None
        # 1) direct internal key
        if feat in user_dict:
            val = user_dict.get(feat)
        else:
            # 2) check readable aliases
            aliases = internal_to_readables.get(feat, [])
            for a in aliases:
                if a in user_dict:
                    val = user_dict.get(a)
                    break
        if val is None:
            missing.append(feat)
            val = 0.0
        try:
            row.append(float(val))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Feature '{feat}' has non-numeric value: {val}")

    x = np.array(row).reshape(1, -1)
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {str(e)}")

    try:
        pred = model.predict(x_scaled)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    return {"Predicted_RUL": float(pred), "missing_filled_with_zero": missing}
