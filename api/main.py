from fastapi import FastAPI
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
#from pydantic import BaseModel

from main.inference import inference

api = FastAPI(title="Ranking (RecSys) API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/recommend")
def recommend(user_id: int):
    """Top-3 recommendations for a single user_id."""
    results = inference(user_id=user_id)

    if results.empty:
        return {"message": f"No recommendations found for user {user_id}"}

    return results.to_dict(orient="records")

@api.get("/")
def root():
    return {"message": "Welcome to the Ranking (RecSys) API!"}

@api.get("/health")
def health_check():
    try:
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "details": str(e)}
