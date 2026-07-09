# /home/simon/otitenet/otitenet/api/main.py

import argparse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import List

from otitenet.api.mobile_deployment import (
    get_current_deployment_manifest,
    get_deployment_file_path,
)
from otitenet.app.analysis import run_analysis_on_file
from otitenet.app.database import create_db, get_production_model
from otitenet.app.services.production_model_service import apply_production_model_to_args

app = FastAPI(title="OtiteNet Mobile API")


@app.get("/")
def root():
    return {
        "status": "ok",
        "name": "OtiteNet Mobile API",
        "health": "/health",
        "current_deployment": "/deployment/current",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/deployment/current")
def deployment_current():
    """
    Android calls this endpoint to know which model is currently deployed.
    The web admin controls this by updating data/mobile_deployments/current/manifest.json.
    """
    manifest = get_current_deployment_manifest()

    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail="No active mobile deployment found.",
        )

    return manifest


@app.get("/deployment/current/files/{filename}")
def deployment_current_file(filename: str):
    """
    Android downloads model/assets from here.
    """
    path = get_deployment_file_path(filename)

    if path is None or not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Deployment file not found: {filename}",
        )

    return FileResponse(path)


@app.post("/analyze")
async def analyze_image(
    person_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    Android uploads an image here to get an immediate prediction.
    Uses the production model set by the admin in the web app.
    """
    conn, cursor = create_db()
    prod_model = get_production_model(cursor)

    if not prod_model:
        raise HTTPException(status_code=400, detail="No production model set in admin panel.")

    # Initialize args and apply production model settings
    # These defaults match the common configuration
    args = argparse.Namespace(
        device="cpu",
        task="otite",
        path="data/otite_ds_64",
        new_size=64,
        normalize="yes",
        n_neighbors=1,
        dist_fct="euclidean",
        classif_loss="triplet",
        dloss="triplet",
        prototypes_to_use="class",
        n_positives="1",
        n_negatives="1",
        fgsm="0",
        n_calibration="0"
    )
    args = apply_production_model_to_args(args, prod_model)

    # Note: run_analysis_on_file expects st.session_state.person_id
    # We need to monkeypatch or ensure database.py is flexible.
    # For the API, we'll inject it into a mock session state if needed,
    # but run_analysis_on_file usually handles the DB insert.
    import streamlit as st
    if 'person_id' not in st.session_state:
        st.session_state.person_id = person_id

    image_bytes = await file.read()

    try:
        # result: (pred_label, confidence, log_path, existing, gradcam_path)
        result = run_analysis_on_file(
            file.filename,
            image_bytes,
            args,
            cursor,
            conn,
            force_reanalyze=True # Mobile usually wants a fresh check or latest production
        )

        return {
            "prediction": result[0],
            "confidence": float(result[1]),
            "filename": file.filename,
            "timestamp": getattr(args, "timestamp", None) # Optional
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{person_id}")
async def get_results(person_id: int):
    """
    Returns the history of analyses for a specific family member.
    """
    conn, cursor = create_db()
    cursor.execute(
        """
        SELECT filename, pred_label, confidence, timestamp
        FROM results
        WHERE person_id=%s
        ORDER BY timestamp DESC
        """,
        (person_id,)
    )
    rows = cursor.fetchall()
    return [
        {
            "filename": r[0],
            "prediction": r[1],
            "confidence": float(r[2]) if r[2] is not None else 0.0,
            "timestamp": str(r[3])
        }
        for r in rows
    ]
