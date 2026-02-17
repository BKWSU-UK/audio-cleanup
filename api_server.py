"""
API Server for the Intelligent Studio Pipeline.

Provides REST endpoints for uploading audio files, processing them through
the pipeline (synchronously or asynchronously), and downloading results.

Run with: python api_server.py [--host 0.0.0.0] [--port 8000]
Docs at:  http://localhost:8000/docs
"""

import os
import uuid
import shutil
import logging
import tempfile
import traceback
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from pipeline import PipelineConfig, IntelligentStudioPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# Job storage
# ---------------------------------------------------------------------------
JOBS_DIR = Path(tempfile.gettempdir()) / "audio-cleanup-jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_format: str = "opus"
    download_url: Optional[str] = None


# In-memory job registry (sufficient for single-instance use)
jobs: dict[str, JobInfo] = {}

# Thread pool for async processing (1 worker — pipeline is CPU/GPU heavy)
executor = ThreadPoolExecutor(max_workers=1)

# Shared pipeline instances keyed by config hash to avoid re-loading the AI model
_pipeline_cache: dict[str, IntelligentStudioPipeline] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_config(
    no_preprocessing: bool = False,
    no_ai: bool = False,
    no_spectral_denoise: bool = False,
    no_noise_gate: bool = False,
    no_mastering: bool = False,
    output_format: str = "opus",
    bitrate: int = 64,
    noise_reduction: float = PipelineConfig.NOISE_REDUCTION_DB,
    noise_sensitivity: float = PipelineConfig.NOISE_SENSITIVITY,
    gate_threshold: float = PipelineConfig.GATE_THRESHOLD_DB,
    gate_attack: float = PipelineConfig.GATE_ATTACK_MS,
    gate_release: float = PipelineConfig.GATE_RELEASE_MS,
    lowpass_threshold: float = PipelineConfig.LOW_PASS_THRESHOLD,
    lowpass_cutoff: int = PipelineConfig.LOW_PASS_CUTOFF,
    speechnorm_expansion: float = PipelineConfig.SPEECHNORM_EXPANSION,
    highpass: int = PipelineConfig.HIGHPASS_FREQ,
    loudness: float = PipelineConfig.LOUDNESS_TARGET,
    true_peak: float = PipelineConfig.TRUE_PEAK,
) -> PipelineConfig:
    """Build a PipelineConfig from API parameters."""
    cfg = PipelineConfig()
    cfg.ENABLE_PREPROCESSING = not no_preprocessing
    cfg.ENABLE_AI_ENHANCEMENT = not no_ai
    cfg.ENABLE_SPECTRAL_DENOISE = not no_spectral_denoise
    cfg.ENABLE_NOISE_GATE = not no_noise_gate
    cfg.ENABLE_MASTERING = not no_mastering
    cfg.OUTPUT_FORMAT = output_format
    cfg.OUTPUT_BITRATE = bitrate
    cfg.NOISE_REDUCTION_DB = noise_reduction
    cfg.NOISE_SENSITIVITY = noise_sensitivity
    cfg.GATE_THRESHOLD_DB = gate_threshold
    cfg.GATE_ATTACK_MS = gate_attack
    cfg.GATE_RELEASE_MS = gate_release
    cfg.LOW_PASS_THRESHOLD = lowpass_threshold
    cfg.LOW_PASS_CUTOFF = lowpass_cutoff
    cfg.SPEECHNORM_EXPANSION = speechnorm_expansion
    cfg.HIGHPASS_FREQ = highpass
    cfg.LOUDNESS_TARGET = loudness
    cfg.TRUE_PEAK = true_peak
    return cfg


def _config_cache_key(cfg: PipelineConfig) -> str:
    """Key that captures only the flag that affects model loading."""
    return f"ai={cfg.ENABLE_AI_ENHANCEMENT}"


def _get_pipeline(cfg: PipelineConfig) -> IntelligentStudioPipeline:
    """Return a (possibly cached) pipeline instance."""
    key = _config_cache_key(cfg)
    if key not in _pipeline_cache:
        logger.info("Initialising pipeline (%s) — this may take a moment on first run…", key)
        _pipeline_cache[key] = IntelligentStudioPipeline(cfg)
    return _pipeline_cache[key]


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _run_pipeline(job_id: str, cfg: PipelineConfig):
    """Execute the pipeline for a job. Runs in the thread pool."""
    job = jobs[job_id]
    job.status = JobStatus.processing
    jdir = _job_dir(job_id)
    input_file = jdir / "input" / job.filename
    output_file = jdir / "output" / f"{Path(job.filename).stem}.{cfg.OUTPUT_FORMAT}"

    try:
        pipeline = _get_pipeline(cfg)
        # Override per-job config values (model is shared, but processing params differ)
        pipeline.config = cfg
        pipeline.process_file(str(input_file), str(output_file))

        if not output_file.exists():
            raise RuntimeError("Pipeline completed but output file was not created")

        job.status = JobStatus.completed
        job.completed_at = datetime.now(timezone.utc).isoformat()
        job.download_url = f"/jobs/{job_id}/download"
        logger.info("Job %s completed: %s", job_id, output_file.name)

    except Exception as exc:
        job.status = JobStatus.failed
        job.completed_at = datetime.now(timezone.utc).isoformat()
        job.error = f"{type(exc).__name__}: {exc}"
        logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Audio Cleanup Pipeline API",
    description="Upload audio → AI-powered enhancement → download the result. "
                "Supports synchronous and asynchronous processing.",
    version="1.0.0",
)


# ---- Common query parameters (reused by both sync and async endpoints) ----
PARAM_DOCS = {
    "no_preprocessing": "Disable preprocessing (channel selection, clipping, normalization)",
    "no_ai": "Disable AI enhancement (ClearVoice)",
    "no_spectral_denoise": "Disable spectral noise reduction",
    "no_noise_gate": "Disable noise gate",
    "no_mastering": "Disable mastering (EQ, de-essing, loudness normalization)",
    "output_format": "Output format",
    "bitrate": "Opus bitrate in kbps (ignored for FLAC)",
    "noise_reduction": "Spectral noise reduction amount in dB",
    "noise_sensitivity": "Noise detection sensitivity (0-1, lower = more aggressive)",
    "gate_threshold": "Noise gate threshold in dB FS",
    "gate_attack": "Noise gate attack time in ms",
    "gate_release": "Noise gate release time in ms",
    "lowpass_threshold": "Threshold for detecting poor HF content (0-1)",
    "lowpass_cutoff": "Low-pass filter cutoff in Hz",
    "speechnorm_expansion": "Speech normalization expansion factor",
    "highpass": "Highpass filter frequency in Hz",
    "loudness": "Target loudness in LUFS",
    "true_peak": "True peak limit in dB",
}


def _save_upload(file: UploadFile, job_id: str) -> str:
    """Save uploaded file to job input directory. Returns the filename."""
    jdir = _job_dir(job_id)
    (jdir / "input").mkdir(parents=True, exist_ok=True)
    (jdir / "output").mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"upload_{job_id}"
    dest = jdir / "input" / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return filename


# ---------------------------------------------------------------------------
# POST /process  — synchronous: upload, process, return file
# ---------------------------------------------------------------------------
@app.post(
    "/process",
    summary="Process audio synchronously",
    description="Upload an audio file, process it through the pipeline, and receive "
                "the enhanced file directly in the response. Blocks until complete.",
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def process_sync(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to process"),
    no_preprocessing: bool = Query(False, description=PARAM_DOCS["no_preprocessing"]),
    no_ai: bool = Query(False, description=PARAM_DOCS["no_ai"]),
    no_spectral_denoise: bool = Query(False, description=PARAM_DOCS["no_spectral_denoise"]),
    no_noise_gate: bool = Query(False, description=PARAM_DOCS["no_noise_gate"]),
    no_mastering: bool = Query(False, description=PARAM_DOCS["no_mastering"]),
    output_format: str = Query("opus", enum=["opus", "flac"], description=PARAM_DOCS["output_format"]),
    bitrate: int = Query(64, description=PARAM_DOCS["bitrate"]),
    noise_reduction: float = Query(PipelineConfig.NOISE_REDUCTION_DB, description=PARAM_DOCS["noise_reduction"]),
    noise_sensitivity: float = Query(PipelineConfig.NOISE_SENSITIVITY, description=PARAM_DOCS["noise_sensitivity"]),
    gate_threshold: float = Query(PipelineConfig.GATE_THRESHOLD_DB, description=PARAM_DOCS["gate_threshold"]),
    gate_attack: float = Query(PipelineConfig.GATE_ATTACK_MS, description=PARAM_DOCS["gate_attack"]),
    gate_release: float = Query(PipelineConfig.GATE_RELEASE_MS, description=PARAM_DOCS["gate_release"]),
    lowpass_threshold: float = Query(PipelineConfig.LOW_PASS_THRESHOLD, description=PARAM_DOCS["lowpass_threshold"]),
    lowpass_cutoff: int = Query(PipelineConfig.LOW_PASS_CUTOFF, description=PARAM_DOCS["lowpass_cutoff"]),
    speechnorm_expansion: float = Query(PipelineConfig.SPEECHNORM_EXPANSION, description=PARAM_DOCS["speechnorm_expansion"]),
    highpass: int = Query(PipelineConfig.HIGHPASS_FREQ, description=PARAM_DOCS["highpass"]),
    loudness: float = Query(PipelineConfig.LOUDNESS_TARGET, description=PARAM_DOCS["loudness"]),
    true_peak: float = Query(PipelineConfig.TRUE_PEAK, description=PARAM_DOCS["true_peak"]),
):
    job_id = str(uuid.uuid4())
    cfg = _build_config(
        no_preprocessing=no_preprocessing, no_ai=no_ai,
        no_spectral_denoise=no_spectral_denoise, no_noise_gate=no_noise_gate,
        no_mastering=no_mastering, output_format=output_format, bitrate=bitrate,
        noise_reduction=noise_reduction, noise_sensitivity=noise_sensitivity,
        gate_threshold=gate_threshold, gate_attack=gate_attack, gate_release=gate_release,
        lowpass_threshold=lowpass_threshold, lowpass_cutoff=lowpass_cutoff,
        speechnorm_expansion=speechnorm_expansion, highpass=highpass,
        loudness=loudness, true_peak=true_peak,
    )

    filename = _save_upload(file, job_id)
    jobs[job_id] = JobInfo(
        job_id=job_id, status=JobStatus.pending, filename=filename,
        created_at=datetime.now(timezone.utc).isoformat(), output_format=output_format,
    )

    _run_pipeline(job_id, cfg)
    job = jobs[job_id]

    if job.status == JobStatus.failed:
        _cleanup_job(job_id)
        raise HTTPException(status_code=500, detail=job.error)

    output_file = _job_dir(job_id) / "output" / f"{Path(filename).stem}.{output_format}"
    media_type = "audio/flac" if output_format == "flac" else "audio/ogg"

    background_tasks.add_task(_cleanup_job, job_id)
    return FileResponse(
        path=str(output_file),
        media_type=media_type,
        filename=output_file.name,
        background=background_tasks,
    )


# ---------------------------------------------------------------------------
# POST /jobs  — asynchronous: upload and start processing
# ---------------------------------------------------------------------------
@app.post(
    "/jobs",
    response_model=JobInfo,
    status_code=202,
    summary="Submit audio for async processing",
    description="Upload an audio file and start processing in the background. "
                "Returns a job ID to poll for status.",
)
def submit_job(
    file: UploadFile = File(..., description="Audio file to process"),
    no_preprocessing: bool = Query(False, description=PARAM_DOCS["no_preprocessing"]),
    no_ai: bool = Query(False, description=PARAM_DOCS["no_ai"]),
    no_spectral_denoise: bool = Query(False, description=PARAM_DOCS["no_spectral_denoise"]),
    no_noise_gate: bool = Query(False, description=PARAM_DOCS["no_noise_gate"]),
    no_mastering: bool = Query(False, description=PARAM_DOCS["no_mastering"]),
    output_format: str = Query("opus", enum=["opus", "flac"], description=PARAM_DOCS["output_format"]),
    bitrate: int = Query(64, description=PARAM_DOCS["bitrate"]),
    noise_reduction: float = Query(PipelineConfig.NOISE_REDUCTION_DB, description=PARAM_DOCS["noise_reduction"]),
    noise_sensitivity: float = Query(PipelineConfig.NOISE_SENSITIVITY, description=PARAM_DOCS["noise_sensitivity"]),
    gate_threshold: float = Query(PipelineConfig.GATE_THRESHOLD_DB, description=PARAM_DOCS["gate_threshold"]),
    gate_attack: float = Query(PipelineConfig.GATE_ATTACK_MS, description=PARAM_DOCS["gate_attack"]),
    gate_release: float = Query(PipelineConfig.GATE_RELEASE_MS, description=PARAM_DOCS["gate_release"]),
    lowpass_threshold: float = Query(PipelineConfig.LOW_PASS_THRESHOLD, description=PARAM_DOCS["lowpass_threshold"]),
    lowpass_cutoff: int = Query(PipelineConfig.LOW_PASS_CUTOFF, description=PARAM_DOCS["lowpass_cutoff"]),
    speechnorm_expansion: float = Query(PipelineConfig.SPEECHNORM_EXPANSION, description=PARAM_DOCS["speechnorm_expansion"]),
    highpass: int = Query(PipelineConfig.HIGHPASS_FREQ, description=PARAM_DOCS["highpass"]),
    loudness: float = Query(PipelineConfig.LOUDNESS_TARGET, description=PARAM_DOCS["loudness"]),
    true_peak: float = Query(PipelineConfig.TRUE_PEAK, description=PARAM_DOCS["true_peak"]),
):
    job_id = str(uuid.uuid4())
    cfg = _build_config(
        no_preprocessing=no_preprocessing, no_ai=no_ai,
        no_spectral_denoise=no_spectral_denoise, no_noise_gate=no_noise_gate,
        no_mastering=no_mastering, output_format=output_format, bitrate=bitrate,
        noise_reduction=noise_reduction, noise_sensitivity=noise_sensitivity,
        gate_threshold=gate_threshold, gate_attack=gate_attack, gate_release=gate_release,
        lowpass_threshold=lowpass_threshold, lowpass_cutoff=lowpass_cutoff,
        speechnorm_expansion=speechnorm_expansion, highpass=highpass,
        loudness=loudness, true_peak=true_peak,
    )

    filename = _save_upload(file, job_id)
    job = JobInfo(
        job_id=job_id, status=JobStatus.pending, filename=filename,
        created_at=datetime.now(timezone.utc).isoformat(), output_format=output_format,
    )
    jobs[job_id] = job

    executor.submit(_run_pipeline, job_id, cfg)
    logger.info("Job %s queued: %s", job_id, filename)

    return job


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}  — poll job status
# ---------------------------------------------------------------------------
@app.get(
    "/jobs/{job_id}",
    response_model=JobInfo,
    summary="Get job status",
    description="Poll the status of an async processing job. "
                "When status is 'completed', a download_url will be provided.",
)
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/download  — download completed result
# ---------------------------------------------------------------------------
@app.get(
    "/jobs/{job_id}/download",
    summary="Download processed audio",
    description="Download the processed audio file for a completed job.",
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def download_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status == JobStatus.failed:
        raise HTTPException(status_code=500, detail=job.error)
    if job.status in (JobStatus.pending, JobStatus.processing):
        raise HTTPException(status_code=409, detail=f"Job is still {job.status.value}")

    output_file = _job_dir(job_id) / "output" / f"{Path(job.filename).stem}.{job.output_format}"
    if not output_file.exists():
        raise HTTPException(status_code=500, detail="Output file not found")

    media_type = "audio/flac" if job.output_format == "flac" else "audio/ogg"
    return FileResponse(path=str(output_file), media_type=media_type, filename=output_file.name)


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id}  — clean up a job
# ---------------------------------------------------------------------------
@app.delete(
    "/jobs/{job_id}",
    summary="Delete a job and its files",
    description="Remove a job and clean up its temporary files.",
)
def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status == JobStatus.processing:
        raise HTTPException(status_code=409, detail="Cannot delete a job that is still processing")

    _cleanup_job(job_id)
    return {"detail": "Job deleted"}


# ---------------------------------------------------------------------------
# GET /jobs  — list all jobs
# ---------------------------------------------------------------------------
@app.get(
    "/jobs",
    response_model=list[JobInfo],
    summary="List all jobs",
)
def list_jobs():
    return list(jobs.values())


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def _cleanup_job(job_id: str):
    jdir = _job_dir(job_id)
    if jdir.exists():
        shutil.rmtree(jdir, ignore_errors=True)
    jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Audio Cleanup Pipeline API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=args.reload)
