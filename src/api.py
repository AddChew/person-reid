import torch
import logging

import numpy as np
import pandas as pd

from typing import Literal
from pydantic import BaseModel
from utils import FeatureExtractor

from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s")
model_artifacts = {}


class Message(BaseModel):
    """
    Heartbeat message schema.
    """
    message: str = Literal["pong"]


class Infer(BaseModel):
    """
    Infer schema.
    """
    cluster: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts in lifespan event.

    Args:
        app (FastAPI): FastAPI app.
    """
    logging.info("Load feature extractor")
    model_artifacts["feature_extractor"] = FeatureExtractor(
        model_name = 'osnet_x0_25',
        model_path = 'models/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth',
        device = 'cpu'
    )

    logging.info("Load embeddings")
    model_artifacts["embeddings"] = np.load("outputs/embeddings.npy")

    logging.info("Load labels")
    labels = pd.read_csv("outputs/Addison_clusterid.csv")
    model_artifacts["labels"] = dict(zip(labels.index.values, labels.ClusterID))

    yield

    logging.info("Clear model artifacts")
    model_artifacts.clear()


app = FastAPI(
    title = "Person Re-Identification",
    description = "Model Serving Endpoint Documentation for Person Re-Identification Service.",
    lifespan = lifespan,
)


@app.get("/ping")
async def ping() -> Message:
    """
    Endpoint to poll if the service is alive and running
    """
    return {"message": "pong"}


@app.post("/infer")
async def infer(files: list[UploadFile]) -> Infer:
    """
    Endpoint to run inference on uploaded file.
    """
    logging.info("Extract embedding")
    files = [file.file for file in files]
    with torch.no_grad():
        embedding = model_artifacts["feature_extractor"](files)
    logging.info(f"Embedding shape: {embedding.shape}")

    logging.info("Compute cosine similarities")
    scores = cosine_similarity(embedding, model_artifacts["embeddings"])

    logging.info("Find nearest neighbour")
    idx = np.argmax(scores)
    return {"cluster": model_artifacts["labels"].get(idx, -1)}