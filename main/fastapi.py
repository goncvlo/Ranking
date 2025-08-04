from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
import pandas as pd

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
    try:
        results = inference(user_id=user_id)

        if results is None:
            return {"message": "Inference returned None"}

        if results.empty:
            return {"message": f"No recommendations found for user {user_id}"}

        return results.to_dict(orient="records")

    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


@api.get("/")
def root():
    return {"message": "Welcome to the Ranking (RecSys) API!"}


@api.get("/health")
def health_check():
    try:
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "details": str(e)}
