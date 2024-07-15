import numpy as np

from typing import Literal
from fastapi import FastAPI

from pydantic import BaseModel
from src.utils import FeatureExtractor


class Message(BaseModel):
    message: str = Literal["pong"]


app = FastAPI(
    title = "Person Re-Identification",
    description = "Model Serving Endpoint Documentation for Person Re-Identification Service.",
)


@app.get("/ping")
async def ping() -> Message:
    """
    Endpoint to poll if the service is alive and running
    """
    return {"message": "pong"}


# from sklearn.metrics.pairwise import cosine_similarity

# scores = cosine_similarity(features.numpy(), embeddings)

# import numpy as np

# np.argmax(scores)

# embeddings = np.load("outputs/embeddings.npy")