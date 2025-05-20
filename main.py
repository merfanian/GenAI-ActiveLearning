import logging
from fastapi import FastAPI
import uvicorn
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR
from routers.orchestration_router import router

logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directories ensured: {AUGMENTED_IMAGES_DIR}, {TRAINED_MODELS_DIR}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)