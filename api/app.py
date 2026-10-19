"""Credit default risk API.

Runs locally with:   uvicorn api.app:app --reload    (from the project root)
Runs on AWS Lambda through the `handler` adapter at the bottom.

Endpoints
  GET  /health   -> status check
  GET  /schema   -> form fields (labels, ranges, options, defaults) for the front end
  POST /predict  -> default probability, decision and reason codes for one applicant
"""
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from features import to_model_frame

# In the container, models/ sits next to app.py; locally it's in the project root.
_here = Path(__file__).resolve().parent
MODEL_DIR = _here / "models" if (_here / "models").exists() else _here.parent / "models"
SCHEMA = json.loads((MODEL_DIR / "slim_schema.json").read_text())
BOOSTER = xgb.Booster()
BOOSTER.load_model(MODEL_DIR / "slim_model.json")
FIELDS = {f["name"]: f for f in SCHEMA["form_fields"]}
THRESHOLD = SCHEMA["threshold"]

# Plain-English names for reason codes.
LABELS = {f["name"]: f["label"] for f in SCHEMA["form_fields"]}
LABELS.update({
    "EXT_SOURCE_MEAN": "Average credit bureau score",
    "EXT_SOURCE_MIN": "Lowest credit bureau score",
    "CREDIT_TO_GOODS": "Loan amount vs. price of goods",
    "CREDIT_TERM": "Repayment speed (annual repayment / loan)",
    "ANNUITY_INCOME_RATIO": "Share of income going to repayments",
    "CREDIT_INCOME_RATIO": "Loan size vs. income",
    "EMPLOYED_YEARS": "Years in current job",
    "DAYS_EMPLOYED_ANOM": "Not in standard employment",
})

app = FastAPI(title="Credit Default Risk API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

Value = Optional[Union[float, int, str]]


@app.get("/health")
def health():
    return {"status": "ok", "model": SCHEMA["model"], "test_auc": SCHEMA["test_auc"]}


@app.get("/schema")
def schema():
    return {"fields": SCHEMA["form_fields"], "threshold": THRESHOLD,
            "excluded_on_fairness_grounds": SCHEMA["excluded_on_fairness_grounds"]}


def _to_raw_row(payload: dict) -> pd.DataFrame:
    """Validate the request against the form spec and convert it to the raw training format."""
    unknown = set(payload) - set(FIELDS)
    if unknown:
        raise HTTPException(422, f"Unknown fields: {sorted(unknown)}")
    row = {}
    for name, f in FIELDS.items():
        v = payload.get(name)
        if v is None or v == "":
            if f.get("required"):
                raise HTTPException(422, f"Missing required field: {name} ({f['label']})")
            row[name] = np.nan
            continue
        if f["type"] == "category":
            if str(v) not in f["options"]:
                raise HTTPException(422, f"{name} must be one of {f['options']}")
            row[name] = str(v)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise HTTPException(422, f"{name} must be a number")
        if v < 0:
            raise HTTPException(422, f"{name} cannot be negative")
        if f["type"] == "score" and v > 1:
            raise HTTPException(422, f"{name} must be between 0 and 1")
        if f["type"] == "years":  # the model was trained on negative days before application
            v = -v * 365.25
        row[name] = v
    return pd.DataFrame([row])


@app.post("/predict")
def predict(payload: dict[str, Value]):
    X = to_model_frame(_to_raw_row(payload), SCHEMA)
    dm = xgb.DMatrix(X, enable_categorical=True)
    margin = float(BOOSTER.predict(dm, output_margin=True)[0])
    prob = 1.0 / (1.0 + np.exp(-margin))
    contribs = pd.Series(BOOSTER.predict(dm, pred_contribs=True)[0, :-1], index=X.columns)

    def reasons(series):
        return [{"feature": f, "label": LABELS.get(f, f), "impact": round(float(v), 3)} for f, v in series.items()]

    return {
        "default_probability": round(prob, 4),
        "decision": "APPROVE" if prob < THRESHOLD else "DECLINE",
        "threshold": THRESHOLD,
        "raised_risk": reasons(contribs[contribs > 0].sort_values(ascending=False).head(3)),
        "lowered_risk": reasons(contribs[contribs < 0].sort_values().head(3)),
    }


handler = Mangum(app)  # AWS Lambda entry point
