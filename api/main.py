from fastapi import FastAPI, Query
from typing import List
import pandas as pd

from main.inference import inference

api = FastAPI(title="Recommendation API")

@api.get("/recommend")
def recommend(user_ids: List[int] = Query(...)):
    """
    Get top-N recommendations for a list of user_ids.
    Example: /recommend?user_ids=101&user_ids=202&top_n=5
    """
    results = inference(user_ids=user_ids)
    return results.to_dict(orient="records")

@api.get("/")
def root():
    return {"message": "Welcome to the Recommendation API!"}

@api.get("/health")
def health_check():
    try:
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "details": str(e)}
    