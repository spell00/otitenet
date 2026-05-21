# /home/simon/otitenet/otitenet/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from otitenet.api.mobile_deployment import (
    get_current_deployment_manifest,
    get_deployment_file_path,
)

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
    Example:
        /deployment/current/files/model.ptl
        /deployment/current/files/labels.json
        /deployment/current/files/reference_embeddings.npy
    """
    path = get_deployment_file_path(filename)

    if path is None or not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Deployment file not found: {filename}",
        )

    return FileResponse(path)