# Predictive Maintenance API (XGBoost)

This repository provides a REST API that predicts **Remaining Useful Life (RUL)** of engines using an XGBoost model trained on the NASA Turbofan dataset.

## Run Locally

```bash
pip install -r api/requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

## Example Request

POST `/predict`
```json
{
  "data": {
    "op_setting1": 0.8,
    "op_setting2": 518.67,
    "op_setting3": 643.02,
    "s1": 1585.29,
    "s2": 1398.21,
    "s3": 8138.62
  }
}
```

Response:
```json
{"Predicted_RUL": 134.56}
```

## Deployment

- Push this folder to GitHub.
- Connect your repo to [Railway](https://railway.app).
- Railway will detect the Dockerfile and deploy automatically.
- The API will be live at `https://your-app-name.up.railway.app/predict`.
